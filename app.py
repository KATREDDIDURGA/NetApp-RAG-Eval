"""
Streamlit ChatGPT-style Interface for RAG System
Modern chat interface with streaming responses, citations, and debug info
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Direct imports (if running standalone)
DIRECT_MODE = False
system_components = None

try:
    from config import config
    from processor import DocumentProcessor
    from retriever import HybridRetriever
    from agent import RAGAgent
    DIRECT_MODE = True
except ImportError as e:
    st.error(f"Could not import project modules: {e}")
    DIRECT_MODE = False

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #ff6b6b;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #4ecdc4;
        border: 1px solid #e1e5e9;
    }
    
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-content {
        line-height: 1.6;
    }
    
    /* Citations styling */
    .citations {
        margin-top: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .citation-item {
        margin: 0.2rem 0;
        color: #666;
    }
    
    /* Metrics styling */
    .metrics-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        min-width: 120px;
    }
    
    /* Sidebar styling */
    .sidebar .block-container {
        padding-top: 1rem;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if "api_endpoint" not in st.session_state:
    st.session_state.api_endpoint = "http://localhost:8000"

if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}

if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None

# API functions
def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Call the FastAPI backend"""
    try:
        url = f"{st.session_state.api_endpoint}/{endpoint}"
        
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Please ensure the FastAPI server is running on localhost:8000"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be processing a large request."}
    except Exception as e:
        return {"error": f"API error: {str(e)}"}

def check_system_health() -> Dict:
    """Check if the system is healthy"""
    return call_api("health")

def ingest_documents(force_reprocess: bool = False) -> Dict:
    """Trigger document ingestion"""
    return call_api("ingest", "POST", {
        "force_reprocess": force_reprocess
    })

def query_rag(query: str, mode: str = "auto", include_debug: bool = False) -> Dict:
    """Query the RAG system"""
    return call_api("query", "POST", {
        "query": query,
        "mode": mode,
        "include_debug": include_debug
    })

# Direct mode functions (if running without API)
@st.cache_resource
def init_direct_system():
    """Initialize system components directly with proper error handling"""
    if not DIRECT_MODE:
        return None
    
    try:
        st.info("Initializing RAG system in direct mode...")
        
        # Validate config first
        if not hasattr(config, 'processed_data_dir') or not config.processed_data_dir:
            raise ValueError("Config not properly initialized - processed_data_dir missing")
        
        # Initialize processor with proper Path handling
        processor = DocumentProcessor(
            processed_data_dir=config.processed_data_dir,
            processed_docs_file=config.processed_docs_file
        )
        
        # Initialize retriever
        retriever = HybridRetriever(config)
        
        # Initialize agent
        agent = RAGAgent(config, retriever)
        
        st.success("✅ Direct mode initialization successful!")
        
        return {
            "processor": processor, 
            "retriever": retriever, 
            "agent": agent,
            "config": config
        }
        
    except Exception as e:
        error_msg = f"Failed to initialize system: {str(e)}"
        st.error(error_msg)
        st.session_state.initialization_error = error_msg
        
        # Show more detailed error info
        with st.expander("🔍 Detailed Error Information"):
            st.text(f"Error type: {type(e).__name__}")
            st.text(f"Error message: {str(e)}")
            
            # Check if config is available
            if 'config' in globals():
                try:
                    st.text(f"Config available: Yes")
                    st.text(f"Data dir: {getattr(config, 'data_dir', 'Not set')}")
                    st.text(f"Processed data dir: {getattr(config, 'processed_data_dir', 'Not set')}")
                    st.text(f"Raw data dir: {getattr(config, 'raw_data_dir', 'Not set')}")
                except Exception as config_error:
                    st.text(f"Config error: {config_error}")
            else:
                st.text("Config not available")
                
            # Show import status
            st.text(f"DIRECT_MODE: {DIRECT_MODE}")
            
        return None

def query_direct(query: str, system_components: Dict) -> Dict:
    """Query system directly without API"""
    try:
        agent = system_components["agent"]
        result = agent.query(query)
        return result
    except Exception as e:
        return {"error": f"Direct query failed: {e}"}

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    
    if DIRECT_MODE:
        st.success("✅ Direct mode active")
        
        # Try to initialize system components
        if st.session_state.initialization_error:
            st.error("❌ System initialization failed")
            st.error(st.session_state.initialization_error)
            system_components = None
        else:
            system_components = init_direct_system()
            
        if system_components:
            st.success("✅ System initialized")
            st.session_state.system_initialized = True
            
            # Show some basic stats
            try:
                processor = system_components["processor"]
                stats = processor.get_processing_stats()
                st.metric("Documents", stats.get("total_documents", 0))
                st.metric("Chunks", stats.get("total_chunks", 0))
            except Exception as e:
                st.warning(f"Could not load stats: {e}")
        else:
            st.error("❌ System initialization failed")
            st.session_state.system_initialized = False
    else:
        # Check API health
        health = check_system_health()
        if "error" in health:
            st.error("❌ API Server offline")
            st.error(health["error"])
            st.session_state.system_initialized = False
        else:
            st.success("✅ API Server online")
            st.session_state.system_initialized = True
            
            # Show system stats
            uptime = health.get("uptime_seconds", 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            
            st.metric("Uptime", f"{hours}h {minutes}m")
            st.metric("Total Requests", health.get("total_requests", 0))
    
    st.markdown("---")
    
    # Document management
    st.subheader("📚 Knowledge Base")
    
    # Check system status for documents
    docs_processed = False
    doc_count = 0
    
    if DIRECT_MODE and system_components and st.session_state.system_initialized:
        # Direct mode - check processor
        try:
            processor = system_components["processor"]
            stats = processor.get_processing_stats()
            doc_count = stats.get("total_documents", 0)
            docs_processed = doc_count > 0
        except Exception as e:
            st.warning(f"Could not check document status: {e}")
    elif st.session_state.system_initialized and not DIRECT_MODE:
        # API mode - check via health endpoint
        try:
            health = check_system_health()
            if "error" not in health:
                system_info = health.get("system_info", {})
                processing_stats = system_info.get("processing_stats", {})
                doc_count = processing_stats.get("total_documents", 0)
                docs_processed = doc_count > 0
        except Exception as e:
            st.warning(f"Could not check API document status: {e}")
    
    # Show status
    if docs_processed:
        st.success(f"✅ {doc_count} documents ready for search")
    else:
        st.info("📄 Click 'Process' to process your HTML documents")
    
    # Process documents button
    col1, col2 = st.columns([3, 1])
    with col1:
        if docs_processed:
            st.caption("Knowledge base is ready. Process to refresh or add new documents.")
        else:
            st.caption("Process your HTML documents to enable intelligent search")
    with col2:
        button_text = "🔄 Refresh" if docs_processed else "🚀 Process"
        
        if st.button(button_text, use_container_width=True, disabled=not st.session_state.system_initialized):
            if not st.session_state.system_initialized:
                st.error("System not initialized. Cannot process documents.")
            else:
                with st.spinner("Processing documents..."):
                    if DIRECT_MODE and system_components:
                        try:
                            processor = system_components["processor"]
                            config_obj = system_components["config"]
                            
                            # Process documents
                            result = processor.process_directory(
                                config_obj.raw_data_dir, 
                                force_reprocess=True
                            )
                            
                            if result["processed_count"] > 0:
                                st.success(f"✅ Processed {result['processed_count']} documents")
                                
                                # Index documents
                                retriever = system_components["retriever"]
                                retriever.index_documents(result["document_ids"])
                                st.success("✅ Documents indexed and ready")
                                st.rerun()
                            else:
                                st.warning("No new documents to process")
                                
                        except Exception as e:
                            st.error(f"❌ Processing failed: {e}")
                    else:
                        # API mode
                        result = ingest_documents(force_reprocess=True)
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success(f"✅ Processed {result['processed_count']} documents")
                            st.session_state.processing_stats = result
                            st.rerun()
    
    st.markdown("---")
    
    # Query settings
    st.subheader("⚙️ Query Settings")
    
    search_mode = st.selectbox(
        "Search Mode",
        ["auto", "hybrid", "vector", "keyword"],
        help="auto: Let the system decide, hybrid: Combine dense+sparse, vector: Semantic only, keyword: Exact matching"
    )
    
    include_debug = st.checkbox(
        "Show Debug Info",
        help="Include retrieval details and confidence scores"
    )
    
    temperature = st.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = deterministic, 1 = creative"
    )
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 RAG Chat Assistant")

# Show system status in main area if not ready
if not st.session_state.system_initialized:
    st.warning("⚠️ System not ready. Please check the sidebar for initialization status.")
    
    if st.session_state.initialization_error:
        with st.expander("🔧 Troubleshooting"):
            st.error("Initialization Error:")
            st.code(st.session_state.initialization_error)
            st.info("""
            **Common solutions:**
            1. Make sure your .env file contains UPSTAGE_API_KEY
            2. Check that all required dependencies are installed
            3. Verify that the data directory exists
            4. Try restarting the Streamlit app
            """)
    
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">
                    👤 You
                    <span style="font-size: 0.8em; color: #666;">{message.get('timestamp', '')}</span>
                </div>
                <div class="message-content">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    🤖 Assistant
                    <span style="font-size: 0.8em; color: #666;">{message.get('timestamp', '')}</span>
                </div>
                <div class="message-content">{message['content']}</div>
            """, unsafe_allow_html=True)
            
            # Show citations if available
            if message.get('citations'):
                citations_html = "<div class='citations'><strong>📚 Sources:</strong><br>"
                for citation in message['citations']:
                    citations_html += f"<div class='citation-item'>• {citation}</div>"
                citations_html += "</div>"
                st.markdown(citations_html, unsafe_allow_html=True)
            
            # Show debug info if available
            if message.get('debug_info') and include_debug:
                with st.expander("🔍 Debug Information", expanded=False):
                    debug_info = message['debug_info']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{message.get('confidence', 0):.2f}")
                    with col2:
                        st.metric("Mode", message.get('retrieval_mode', 'unknown'))
                    with col3:
                        st.metric("Documents", debug_info.get('document_count', 0))
                    
                    if debug_info.get('critique'):
                        st.json(debug_info['critique'])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents...", disabled=not st.session_state.system_initialized):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Show user message immediately
    with st.container():
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">
                👤 You
                <span style="font-size: 0.8em; color: #666;">{timestamp}</span>
            </div>
            <div class="message-content">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate response
    with st.container():
        # Show loading message
        loading_placeholder = st.empty()
        loading_placeholder.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">
                🤖 Assistant
                <span style="font-size: 0.8em; color: #666;">{timestamp}</span>
            </div>
            <div class="message-content">
                <span class="loading-dots">Thinking</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Query the system
        start_time = time.time()
        
        if DIRECT_MODE and system_components:
            result = query_direct(prompt, system_components)
        else:
            result = query_rag(prompt, search_mode, include_debug)
        
        processing_time = time.time() - start_time
        
        # Clear loading message
        loading_placeholder.empty()
        
        if "error" in result:
            # Error response
            error_msg = result["error"]
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    🤖 Assistant
                    <span style="font-size: 0.8em; color: #666;">{timestamp}</span>
                </div>
                <div class="message-content">
                    ❌ I encountered an error: {error_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ I encountered an error: {error_msg}",
                "timestamp": timestamp
            })
        else:
            # Successful response
            answer = result.get("answer", "I couldn't generate a response.")
            citations = result.get("citations", [])
            confidence = result.get("confidence", 0)
            retrieval_mode = result.get("retrieval_mode", "unknown")
            
            # Display response
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    🤖 Assistant
                    <span style="font-size: 0.8em; color: #666;">{timestamp}</span>
                </div>
                <div class="message-content">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show citations
            if citations:
                citations_html = "<div class='citations'><strong>📚 Sources:</strong><br>"
                for citation in citations:
                    citations_html += f"<div class='citation-item'>• {citation}</div>"
                citations_html += "</div>"
                st.markdown(citations_html, unsafe_allow_html=True)
            
            # Show performance metrics
            if include_debug:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col2:
                    st.metric("Mode", retrieval_mode)
                with col3:
                    st.metric("Response Time", f"{processing_time:.1f}s")
                with col4:
                    if result.get('debug_info'):
                        st.metric("Sources", result['debug_info'].get('document_count', 0))
            
            # Add to session state
            message_data = {
                "role": "assistant",
                "content": answer,
                "timestamp": timestamp,
                "citations": citations,
                "confidence": confidence,
                "retrieval_mode": retrieval_mode,
                "processing_time": processing_time
            }
            
            if include_debug and result.get('debug_info'):
                message_data["debug_info"] = result['debug_info']
            
            st.session_state.messages.append(message_data)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
    "💡 RAG Chat Assistant powered by Upstage Solar & ChromaDB"
    "</div>",
    unsafe_allow_html=True
)