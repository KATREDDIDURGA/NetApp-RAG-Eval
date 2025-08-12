"""
LangGraph Agent Workflow for RAG System
Handles: Router → Retrieval → Synthesis → Critic → Fallback
"""

import json
import re
import logging
from typing import Dict, List, Optional, TypedDict
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from config import Config
from retriever import HybridRetriever

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Typed state for the RAG agent workflow"""
    query: str
    original_query: str
    retrieval_mode: str  # "vector", "keyword", "hybrid"
    documents: List[Dict]
    context: str
    answer: str
    citations: List[str]
    confidence: float
    critique: Dict
    needs_fallback: bool
    trace_id: str
    timestamp: str

class QueryClassification(BaseModel):
    """Query classification output schema"""
    mode: str = Field(description="Retrieval mode: vector, keyword, or hybrid")
    reasoning: str = Field(description="Why this mode was chosen")
    confidence: float = Field(description="Confidence in classification (0-1)")

class RAGAnswer(BaseModel):
    """Structured answer output schema"""
    answer: str = Field(description="The main answer to the query")
    citations: List[str] = Field(description="Document citations in [doc_id:page] or [file_name:page] format")
    confidence: float = Field(description="Answer confidence (0-1)")

class AnswerCritique(BaseModel):
    """Answer quality critique schema"""
    faithfulness_score: float = Field(description="How well answer follows source context (0-1)")
    citation_accuracy: float = Field(description="Accuracy of citations (0-1)")
    completeness_score: float = Field(description="How complete the answer is (0-1)")
    has_pii: bool = Field(description="Whether answer contains PII")
    issues: List[str] = Field(description="List of identified issues")
    overall_quality: float = Field(description="Overall quality score (0-1)")

class RAGAgent:
    """LangGraph-based RAG agent with multi-step workflow"""

    def __init__(self, config: Config, retriever: HybridRetriever):
        self.config = config
        self.retriever = retriever
        self.llm = self._init_llm()
        self.graph = self._build_graph()

    def _init_llm(self):
        """Initialize Upstage LLM"""
        try:
            from langchain_upstage import ChatUpstage
            return ChatUpstage(
                api_key=self.config.upstage_api_key,
                model="solar-1-mini-chat",
                temperature=0.0  # Deterministic responses
            )
        except ImportError:
            # Fallback to OpenAI-compatible endpoint (Upstage API)
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=self.config.upstage_api_key,
                base_url="https://api.upstage.ai/v1/solar",
                model="solar-1-mini-chat",
                temperature=0.0
            )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("router", self.router_node)
        graph.add_node("retrieval", self.retrieval_node)
        graph.add_node("synthesis", self.synthesis_node)
        graph.add_node("critic", self.critic_node)
        graph.add_node("fallback", self.fallback_node)

        # Define edges
        graph.set_entry_point("router")
        graph.add_edge("router", "retrieval")
        graph.add_edge("retrieval", "synthesis")
        graph.add_edge("synthesis", "critic")

        # Conditional edge from critic
        graph.add_conditional_edges(
            "critic",
            self._should_fallback,
            {"fallback": "fallback", "end": END}
        )
        graph.add_edge("fallback", END)

        return graph.compile()

    # -------------------------
    # Router
    # -------------------------
    def router_node(self, state: AgentState) -> AgentState:
        """Classify query and determine retrieval strategy"""

        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a RAG system.
Analyze the user query and determine the best retrieval mode:

- "vector": For semantic/conceptual questions, complex reasoning
- "keyword": For exact matches, names, specific terms, codes
- "hybrid": For mixed queries needing both semantic and exact matching

You MUST respond with valid JSON only. No explanations or additional text.

Example response (JSON only):
{{"mode": "vector", "reasoning": "Conceptual query about machine learning", "confidence": 0.9}}"""),
            ("human", "Classify this query: {query}\n\nRespond with JSON only:")
        ])

        parser = PydanticOutputParser(pydantic_object=QueryClassification)
        chain = router_prompt | self.llm | parser

        try:
            classification = chain.invoke({"query": state["query"]})
            state["retrieval_mode"] = classification.mode
            state["confidence"] = classification.confidence
        except Exception as e:
            # Fallback to hybrid mode
            logger.warning(f"Router fallback: {e}")
            state["retrieval_mode"] = "hybrid"
            state["confidence"] = 0.5

        return state

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents using hybrid search"""

        # Apply query rewriting/expansion if needed
        expanded_query = self._expand_query(state["query"])

        # Retrieve documents based on mode
        documents = self.retriever.search(
            query=expanded_query,
            mode=state["retrieval_mode"],
            k=self.config.final_k
        )

        # Build context from retrieved documents
        context_parts = []
        citations = []

        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {}) or {}
            content = doc.get("content", "") or ""

            # Prefer human-readable source name (file_name) over raw doc_id
            file_name = metadata.get("file_name")
            page = metadata.get("page", "")
            doc_id = metadata.get("doc_id", f"doc_{i}")

            source_tag = file_name if file_name else doc_id
            citation = f"[{source_tag}:{page}]" if page else f"[{source_tag}]"
            citations.append(citation)

            # Add to context with citation
            context_parts.append(f"{citation} {content}")

        state["documents"] = documents
        state["context"] = "\n\n".join(context_parts)
        state["citations"] = citations

        return state

    # -------------------------
    # Synthesis
    # -------------------------
    def synthesis_node(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved context with robust JSON parsing and clamped citations"""

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer ONLY from the given context
2. If context is insufficient, say "I cannot find enough information to answer this question"
3. Include citations in [doc_id:page] or [file_name:page] format for each claim (keep at most 5 unique citations)
4. Be specific and detailed when context supports it
5. Do not add information not present in the context
6. Maintain a professional, informative tone

CONFIDENCE SCORING:
- 0.9-1.0: Context directly answers the question with specific details
- 0.7-0.8: Context partially answers with good relevant information
- 0.5-0.6: Context has some relevant information but gaps exist
- 0.3-0.4: Context has minimal relevant information
- 0.0-0.2: Context doesn't address the question

You MUST respond with valid JSON only. ALL THREE FIELDS ARE REQUIRED.

REQUIRED JSON FORMAT:
{{
    "answer": "Your detailed answer here",
    "citations": ["[doc1:1]", "[doc2:3]"],
    "confidence": 0.85
}}

Do not include any text outside the JSON. Do not omit the confidence field."""
             ),
            ("human",
             """Context:
{context}

Question: {query}

Respond with JSON only (exactly the three required fields).""")
        ])

        parser = PydanticOutputParser(pydantic_object=RAGAnswer)
        chain = synthesis_prompt | self.llm | parser

        def _postprocess(answer_text: str, citations_list: List[str], conf_val: float):
            # Deduplicate & clamp citations to max 5
            seen = set()
            clean_citations = []
            for c in citations_list or []:
                c = str(c).strip()
                if c and c not in seen:
                    seen.add(c)
                    clean_citations.append(c)
                if len(clean_citations) >= 5:
                    break

            # Normalize confidence to [0,1]
            try:
                conf_val = float(conf_val)
            except Exception:
                conf_val = 0.5
            conf_val = max(0.0, min(1.0, conf_val))

            # Fallback answer if empty
            if not answer_text or not answer_text.strip():
                answer_text = "I cannot find enough information to answer this question from the provided context."

            return answer_text, clean_citations, conf_val

        try:
            result = chain.invoke({
                "context": state["context"],
                "query": state["query"]
            })
            ans, cits, conf = _postprocess(result.answer, result.citations, result.confidence)
            state["answer"] = ans
            state["citations"] = cits
            state["confidence"] = conf

        except Exception as e:
            logger.warning(f"Synthesis parsing failed: {e}")

            # Fallback: get raw and try manual JSON extraction
            try:
                chain_no_parser = synthesis_prompt | self.llm
                raw = chain_no_parser.invoke({
                    "context": state["context"],
                    "query": state["query"]
                })
                response_text = getattr(raw, "content", str(raw))

                m = re.search(r'\{.*\}', response_text, re.DOTALL)
                parsed = {}
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = {}

                ans = parsed.get("answer", response_text.strip())
                cits = parsed.get("citations", [])
                conf = parsed.get("confidence", 0.5)
                ans, cits, conf = _postprocess(ans, cits, conf)

                state["answer"] = ans
                state["citations"] = cits
                state["confidence"] = conf

            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
                state["answer"] = "I encountered an error generating the answer. Please try rephrasing your question."
                state["citations"] = []
                state["confidence"] = 0.3
                state["needs_fallback"] = True

        return state

    # -------------------------
    # Critic
    # -------------------------
    def critic_node(self, state: AgentState) -> AgentState:
        """Evaluate answer quality and check for issues"""

        critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality critic for RAG system answers.
Evaluate the answer against the source context for:

1. Faithfulness: Does the answer accurately reflect the context?
2. Citation accuracy: Are citations properly formatted and relevant?
3. Completeness: Does the answer adequately address the question?
4. PII detection: Does the answer contain personal identifiable information?

You MUST respond with valid JSON only matching this exact schema:
{{
    "faithfulness_score": 0.8,
    "citation_accuracy": 0.9,
    "completeness_score": 0.7,
    "has_pii": false,
    "issues": ["list of issues"],
    "overall_quality": 0.8
}}"""),
            ("human", """Context:
{context}

Question: {query}

Answer: {answer}

Citations: {citations}

Respond with JSON only:""")
        ])

        parser = PydanticOutputParser(pydantic_object=AnswerCritique)
        chain = critic_prompt | self.llm | parser

        try:
            critique = chain.invoke({
                "context": state["context"],
                "query": state["query"],
                "answer": state["answer"],
                "citations": state["citations"]
            })

            state["critique"] = critique.dict()

            # Determine if fallback is needed
            min_quality_threshold = 0.6
            state["needs_fallback"] = (
                critique.overall_quality < min_quality_threshold or
                critique.has_pii or
                len(critique.issues) > 2
            )

        except Exception as e:
            # More robust fallback critique
            logger.warning(f"Critic parsing failed: {e}")

            state["critique"] = {
                "faithfulness_score": 0.7,
                "citation_accuracy": 0.8 if state["citations"] else 0.5,
                "completeness_score": 0.6,
                "has_pii": False,
                "issues": ["Critique parsing failed"],
                "overall_quality": 0.6
            }

            # Conservative approach
            answer_length = len(state["answer"])
            has_citations = len(state["citations"]) > 0

            state["needs_fallback"] = (
                answer_length < 50 or
                "error" in state["answer"].lower() or
                (not has_citations and answer_length < 100)
            )

        return state

    # -------------------------
    # Fallback
    # -------------------------
    def fallback_node(self, state: AgentState) -> AgentState:
        """Handle fallback cases with degraded responses"""

        fallback_answer = (
            "I apologize, but I cannot provide a confident answer to your question based on the available documents.\n\n"
            "This could be because:\n"
            "- The information isn't present in the knowledge base\n"
            "- The question requires information from multiple sources\n"
            "- The context is ambiguous or insufficient\n\n"
            "Please try rephrasing your question or being more specific about what you're looking for."
        )

        state["answer"] = fallback_answer
        state["citations"] = []
        state["confidence"] = 0.1
        state["needs_fallback"] = False  # Prevent loops

        return state

    # -------------------------
    # Helpers
    # -------------------------
    def _should_fallback(self, state: AgentState) -> str:
        """Decide whether to use fallback or end"""
        return "fallback" if state.get("needs_fallback", False) else "end"

    def _expand_query(self, query: str) -> str:
        """Simple query expansion (spell check, acronym expansion)"""
        expanded = query.strip().lower()
        acronyms = {
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "nlp": "natural language processing",
            "api": "application programming interface"
        }
        for acronym, expansion in acronyms.items():
            expanded = re.sub(rf"\b{acronym}\b", f"{acronym} {expansion}", expanded)
        return expanded

    # -------------------------
    # Public entry
    # -------------------------
    def query(self, query: str, trace_id: Optional[str] = None) -> Dict:
        """Main entry point for processing queries"""

        initial_state = AgentState(
            query=query,
            original_query=query,
            retrieval_mode="hybrid",
            documents=[],
            context="",
            answer="",
            citations=[],
            confidence=0.0,
            critique={},
            needs_fallback=False,
            trace_id=trace_id or f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat()
        )

        # Execute the workflow
        final_state = self.graph.invoke(initial_state)

        # Return structured response, including retrieved documents for RAGAS
        return {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "confidence": final_state["confidence"],
            "retrieval_mode": final_state["retrieval_mode"],
            "trace_id": final_state["trace_id"],
            "critique": final_state.get("critique", {}),
            "timestamp": final_state["timestamp"],
            "documents": final_state.get("documents", []),
        }

