"""
Hybrid Retrieval System for RAG
Combines dense embeddings (ChromaDB) with sparse search (BM25)
Includes re-ranking and MMR diversity filtering
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from datetime import datetime
import hashlib

# Vector store
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    logging.error("ChromaDB not available - install with: pip install chromadb")

# BM25 sparse search
try:
    from whoosh.index import create_index, open_dir
    from whoosh.fields import Schema, TEXT, ID, STORED
    from whoosh.qparser import QueryParser
    from whoosh.query import Or, And
    from whoosh import scoring
    import whoosh.analysis as analysis
except ImportError:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logging.warning("No BM25 library available - sparse search disabled")

# Embeddings
try:
    import requests
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("Embedding libraries not available")

# Re-ranking
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    logging.warning("Cross-encoder not available - re-ranking disabled")

from config import Config

logger = logging.getLogger(__name__)


class UpstageEmbeddings:
    """Upstage embedding client with caching"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.upstage_api_key
        self.model = config.embedding_model
        self.cache = {}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with batching"""
        embeddings = []
        
        for i in range(0, len(texts), self.config.embedding_batch_size):
            batch = texts[i:i + self.config.embedding_batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_batch([text])[0]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.cache:
                cached_embeddings.append((i, self.cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self._call_upstage_api(uncached_texts)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    cache_key = hashlib.md5(text.encode()).hexdigest()
                    self.cache[cache_key] = embedding
                
                # Merge cached and new embeddings
                all_embeddings = [None] * len(texts)
                
                # Place cached embeddings
                for i, embedding in cached_embeddings:
                    all_embeddings[i] = embedding
                
                # Place new embeddings
                for i, embedding in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = embedding
                
                return all_embeddings
                
            except Exception as e:
                logger.error(f"Upstage API call failed: {e}")
                # Fallback to sentence transformers
                return self._fallback_embeddings(texts)
        else:
            # All cached
            return [emb for _, emb in sorted(cached_embeddings)]
    
    def _call_upstage_api(self, texts: List[str]) -> List[List[float]]:
        """Call Upstage embeddings API"""
        url = "https://api.upstage.ai/v1/solar/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        for attempt in range(self.config.embedding_max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                
                # L2 normalize if configured
                normalized_embeddings = []
                for embedding in embeddings:
                    embedding = np.array(embedding)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    normalized_embeddings.append(embedding.tolist())
                
                return normalized_embeddings
                
            except Exception as e:
                logger.warning(f"Upstage API attempt {attempt + 1} failed: {e}")
                if attempt < self.config.embedding_max_retries - 1:
                    continue
                raise
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to sentence transformers"""
        try:
            if not hasattr(self, '_fallback_model'):
                self._fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = self._fallback_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Fallback embedding failed: {e}")
            # Return random embeddings as last resort
            return [[0.1] * 384 for _ in texts]


class BM25Retriever:
    """BM25 sparse retrieval using Whoosh or rank_bm25"""
    
    def __init__(self, config: Config):
        self.config = config
        self.use_whoosh = 'whoosh' in globals()
        self.index = None
        self.documents = []
        self.bm25 = None
        
        if self.use_whoosh:
            self._init_whoosh()
        else:
            self._init_rank_bm25()
    
    def _init_whoosh(self):
        """Initialize Whoosh index"""
        try:
            # Define schema
            analyzer = analysis.StemmingAnalyzer() if self.config.use_stemming else analysis.StandardAnalyzer()
            
            schema = Schema(
                doc_id=ID(stored=True),
                chunk_id=ID(stored=True),
                content=TEXT(analyzer=analyzer, stored=True),
                title=TEXT(analyzer=analyzer, stored=True),
                metadata=STORED()
            )
            
            # Create or open index
            index_dir = Path(self.config.chroma_persist_directory) / "bm25_index"
            index_dir.mkdir(exist_ok=True)
            
            try:
                self.index = open_dir(str(index_dir))
                logger.info("Opened existing Whoosh BM25 index")
            except:
                self.index = create_index(schema, str(index_dir))
                logger.info("Created new Whoosh BM25 index")
                
        except Exception as e:
            logger.error(f"Whoosh initialization failed: {e}")
            self.use_whoosh = False
            self._init_rank_bm25()
    
    def _init_rank_bm25(self):
        """Initialize rank_bm25 as fallback"""
        self.documents = []
        self.bm25 = None
        logger.info("Using rank_bm25 for sparse retrieval")
    
    def index_documents(self, documents: List[Dict]):
        """Index documents for BM25 search"""
        if self.use_whoosh:
            self._index_whoosh(documents)
        else:
            self._index_rank_bm25(documents)
    
    def _index_whoosh(self, documents: List[Dict]):
        """Index documents in Whoosh"""
        try:
            writer = self.index.writer()
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                writer.add_document(
                    doc_id=metadata.get("doc_id", ""),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    content=content,
                    title=metadata.get("title", ""),
                    metadata=json.dumps(metadata)
                )
            
            writer.commit()
            logger.info(f"Indexed {len(documents)} documents in Whoosh BM25")
            
        except Exception as e:
            logger.error(f"Whoosh indexing failed: {e}")
    
    def _index_rank_bm25(self, documents: List[Dict]):
        """Index documents in rank_bm25"""
        try:
            self.documents = documents
            
            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                content = doc.get("content", "")
                # Simple tokenization
                tokens = content.lower().split()
                tokenized_docs.append(tokens)
            
            # Create BM25 index
            if 'BM25Okapi' in globals():
                self.bm25 = BM25Okapi(tokenized_docs)
                logger.info(f"Indexed {len(documents)} documents in rank_bm25")
            else:
                logger.warning("BM25Okapi not available")
                
        except Exception as e:
            logger.error(f"rank_bm25 indexing failed: {e}")
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search using BM25"""
        if self.use_whoosh:
            return self._search_whoosh(query, k)
        else:
            return self._search_rank_bm25(query, k)
    
    def _search_whoosh(self, query: str, k: int) -> List[Dict]:
        """Search using Whoosh"""
        try:
            with self.index.searcher(weighting=scoring.BM25F()) as searcher:
                # Parse query
                parser = QueryParser("content", self.index.schema)
                parsed_query = parser.parse(query)
                
                # Search
                results = searcher.search(parsed_query, limit=k)
                
                # Convert to standard format
                documents = []
                for result in results:
                    metadata = json.loads(result["metadata"])
                    documents.append({
                        "content": result["content"],
                        "metadata": metadata,
                        "score": result.score
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Whoosh search failed: {e}")
            return []
    
    def _search_rank_bm25(self, query: str, k: int) -> List[Dict]:
        """Search using rank_bm25"""
        try:
            if not self.bm25 or not self.documents:
                return []
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k documents
            top_indices = np.argsort(scores)[::-1][:k]
            
            documents = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    doc = self.documents[idx].copy()
                    doc["score"] = float(scores[idx])
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"rank_bm25 search failed: {e}")
            return []


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse search"""
    
    def __init__(self, config):
        # Allow either a Config object or a string/path
        if isinstance(config, (str, Path)):
            # Wrap into minimal Config
            temp_config = Config()
            temp_config.chroma_persist_directory = str(config)
            self.config = temp_config
        elif isinstance(config, Config):
            self.config = config
        else:
            raise TypeError(
                f"HybridRetriever expects Config or str/Path, got {type(config)}"
            )

        self.embeddings = UpstageEmbeddings(self.config)
        self.vector_store = None
        self.bm25 = BM25Retriever(self.config)
        self.cross_encoder = None
        self.document_store = {}
        
        self._init_vector_store()
        self._init_cross_encoder()
    
    def _init_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            # Create client
            client = chromadb.PersistentClient(
                path=self.config.chroma_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.vector_store = client.get_collection(
                    name=self.config.chroma_collection_name
                )
                logger.info(f"Loaded existing ChromaDB collection: {self.config.chroma_collection_name}")
            except Exception as e:
                logger.warning(f"Could not load existing collection: {e}")
                
                # Try to delete existing collection and recreate
                try:
                    client.delete_collection(name=self.config.chroma_collection_name)
                    logger.info("Deleted existing collection with incompatible metadata")
                except:
                    pass  # Collection might not exist
                
                # Create new collection with simpler metadata
                embedding_function = embedding_functions.DefaultEmbeddingFunction()
                
                self.vector_store = client.create_collection(
                    name=self.config.chroma_collection_name,
                    embedding_function=embedding_function,
                    metadata={"description": "RAG document collection"}  # Simplified metadata
                )
                logger.info(f"Created new ChromaDB collection: {self.config.chroma_collection_name}")
                
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self.vector_store = None
    
    def _init_cross_encoder(self):
        """Initialize cross-encoder for re-ranking"""
        if not self.config.use_cross_encoder:
            return
        
        try:
            if 'CrossEncoder' in globals():
                self.cross_encoder = CrossEncoder(self.config.cross_encoder_model)
                logger.info(f"Loaded cross-encoder: {self.config.cross_encoder_model}")
            else:
                logger.warning("CrossEncoder not available - re-ranking disabled")
        except Exception as e:
            logger.warning(f"Cross-encoder initialization failed: {e}")
    
    def index_documents(self, document_ids: List[str]):
        """Index processed documents from processor with improved error handling"""
        try:
            # Load documents from processor with correct filename
            processed_file = self.config.processed_data_dir / self.config.processed_docs_file
            
            if not processed_file.exists():
                logger.error(f"No processed documents found at: {processed_file}")
                logger.info(f"Expected file: {processed_file}")
                logger.info(f"Directory contents: {list(self.config.processed_data_dir.iterdir()) if self.config.processed_data_dir.exists() else 'Directory does not exist'}")
                return
            
            logger.info(f"Loading processed documents from: {processed_file}")
            
            with open(processed_file, 'r', encoding='utf-8') as f:
                all_docs = json.load(f)
            
            logger.info(f"Loaded {len(all_docs)} total documents from file")
            
            # Debug: Show available document IDs
            available_doc_ids = list(all_docs.keys())
            logger.info(f"Available document IDs: {available_doc_ids}")
            logger.info(f"Requested document IDs: {document_ids}")
            
            # Collect chunks from specified documents
            chunks_to_index = []
            
            for doc_id in document_ids:
                if doc_id in all_docs:
                    doc_data = all_docs[doc_id]
                    chunks = doc_data.get("chunks", [])
                    file_name = doc_data.get("file_name", "unknown")
                    
                    logger.info(f"Processing document: {file_name} (ID: {doc_id})")
                    logger.info(f"  Found {len(chunks)} chunks in document data")
                    
                    valid_chunks = 0
                    for i, chunk in enumerate(chunks):
                        if chunk and isinstance(chunk, dict) and chunk.get("content"):
                            # Ensure chunk has proper metadata
                            if "metadata" not in chunk:
                                chunk["metadata"] = {}
                            
                            # Add document info to metadata
                            chunk["metadata"]["doc_id"] = doc_id
                            chunk["metadata"]["file_name"] = file_name
                            if "chunk_id" not in chunk["metadata"]:
                                chunk["metadata"]["chunk_id"] = i
                            
                            chunks_to_index.append(chunk)
                            valid_chunks += 1
                            
                            # Debug: Show first chunk content preview
                            if i == 0:
                                content_preview = chunk["content"][:150] + "..."
                                logger.debug(f"  First chunk preview: {content_preview}")
                                
                        else:
                            logger.warning(f"  Skipping invalid chunk {i}: {type(chunk)} - {bool(chunk.get('content') if isinstance(chunk, dict) else False)}")
                    
                    logger.info(f"  Added {valid_chunks} valid chunks to index")
                    
                else:
                    logger.warning(f"Document ID {doc_id} not found in processed documents")
            
            if not chunks_to_index:
                logger.error(f"No valid chunks found to index from {len(document_ids)} documents")
                logger.info("This could be due to:")
                logger.info("1. Document processing failed to create chunks")
                logger.info("2. Chunk data corruption")
                logger.info("3. Mismatch between document IDs")
                return
            
            logger.info(f"Total chunks to index: {len(chunks_to_index)}")
            
            # Index in vector store with detailed logging
            if self.vector_store:
                logger.info("Starting vector store indexing...")
                self._index_vector_store(chunks_to_index)
            else:
                logger.warning("Vector store not available - skipping vector indexing")
            
            # Index in BM25 with detailed logging
            logger.info("Starting BM25 indexing...")
            self.bm25.index_documents(chunks_to_index)
            
            logger.info("✅ Document indexing completed successfully")
            
            # Verify indexing worked
            if self.vector_store:
                try:
                    collection_count = self.vector_store.count()
                    logger.info(f"✅ Vector store now contains {collection_count} documents")
                except Exception as e:
                    logger.warning(f"Could not verify vector store count: {e}")
            
        except FileNotFoundError as e:
            logger.error(f"Processed documents file not found: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse processed documents JSON: {e}")
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _index_vector_store(self, chunks: List[Dict]):
        """Index chunks in ChromaDB with detailed logging"""
        if not self.vector_store:
            logger.warning("Vector store not available")
            return
        
        try:
            logger.info(f"Starting to index {len(chunks)} chunks in vector store...")
            
            # Prepare data
            texts = []
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.get("content"):
                    logger.warning(f"Skipping empty chunk {i}")
                    continue
                    
                content = chunk["content"]
                metadata = chunk.get("metadata", {})
                
                # Clean metadata for ChromaDB compatibility
                cleaned_metadata = self._clean_metadata_for_chroma(metadata)
                
                # Create unique ID
                doc_id = cleaned_metadata.get("doc_id", "unknown")
                chunk_id = cleaned_metadata.get("chunk_id", i)
                unique_id = f"{doc_id}_{chunk_id}_{i}"
                
                texts.append(content)
                ids.append(unique_id)
                metadatas.append(cleaned_metadata)
                
                if i < 3:  # Log first few for debugging
                    logger.info(f"Chunk {i}: ID={unique_id}, content_length={len(content)}, file={cleaned_metadata.get('file_name', 'unknown')}")
            
            if not texts:
                logger.error("No valid texts to index!")
                return
            
            logger.info(f"Prepared {len(texts)} texts for embedding...")
            
            # Get embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}")
            
            # Check if collection exists and clear it
            try:
                existing_count = self.vector_store.count()
                logger.info(f"Collection currently has {existing_count} documents")
                
                if existing_count > 0:
                    logger.info("Clearing existing collection...")
                    # Delete all existing documents
                    existing_docs = self.vector_store.get()
                    if existing_docs['ids']:
                        self.vector_store.delete(ids=existing_docs['ids'])
                        logger.info(f"Deleted {len(existing_docs['ids'])} existing documents")
                        
            except Exception as e:
                logger.warning(f"Could not clear existing collection: {e}")
            
            # Add to ChromaDB in batches
            batch_size = 50  # Smaller batches for reliability
            total_added = 0
            
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                
                logger.info(f"Adding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} items)...")
                
                try:
                    self.vector_store.add(
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    total_added += len(batch_texts)
                    logger.info(f"Successfully added batch {i//batch_size + 1}")
                    
                except Exception as batch_error:
                    logger.error(f"Failed to add batch {i//batch_size + 1}: {batch_error}")
                    # Try individual documents in this batch
                    for j in range(len(batch_texts)):
                        try:
                            self.vector_store.add(
                                documents=[batch_texts[j]],
                                embeddings=[batch_embeddings[j]],
                                ids=[batch_ids[j]],
                                metadatas=[batch_metadatas[j]]
                            )
                            total_added += 1
                        except Exception as doc_error:
                            logger.error(f"Failed to add individual document {batch_ids[j]}: {doc_error}")
            
            # Verify indexing
            final_count = self.vector_store.count()
            logger.info(f"✅ Successfully indexed {total_added} documents. Collection now has {final_count} documents")
            
            # Test a quick search to verify
            if final_count > 0:
                try:
                    # Use embedding-based query instead of text query
                    test_embedding = self.embeddings.embed_query("Hero Wars")
                    test_results = self.vector_store.query(
                        query_embeddings=[test_embedding],
                        n_results=2
                    )
                    logger.info(f"✅ Test search successful: found {len(test_results['documents'][0]) if test_results['documents'] else 0} results")
                except Exception as test_error:
                    logger.warning(f"Test search failed (this is OK): {test_error}")
            
        except Exception as e:
            logger.error(f"Vector store indexing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _clean_metadata_for_chroma(self, metadata: Dict) -> Dict:
        """Clean metadata to be compatible with ChromaDB"""
        cleaned = {}
        
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value and isinstance(value[0], dict):
                    # Handle headings list - extract just the text
                    if key == "headings":
                        text_items = [item.get("text", "") for item in value if isinstance(item, dict)]
                        cleaned[f"{key}_text"] = ", ".join(text_items[:5])  # Limit to first 5
                    else:
                        cleaned[f"{key}_count"] = len(value)
                elif value:
                    cleaned[key] = ", ".join(str(item) for item in value[:5])  # Limit to first 5 items
                else:
                    cleaned[key] = ""
            elif isinstance(value, dict):
                # Convert dicts to JSON strings (truncated)
                json_str = str(value)[:200]  # Limit length
                cleaned[f"{key}_json"] = json_str
            else:
                # Convert other types to string
                cleaned[key] = str(value)[:200]  # Limit length
        
        return cleaned
    
    def search(self, query: str, mode: str = "hybrid", k: int = None) -> List[Dict]:
        """Main search interface with improved relevance scoring"""
        k = k or self.config.final_k
        
        # Enhanced query preprocessing
        enhanced_query = self._enhance_query(query)
        logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
        
        if mode == "vector":
            return self._vector_search(enhanced_query, k)
        elif mode == "keyword":
            return self._sparse_search(enhanced_query, k)
        else:  # hybrid
            return self._hybrid_search(enhanced_query, k)
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query for better retrieval"""
        enhanced = query.lower().strip()
        
        # Extract key entities and concepts
        import re
        
        # Look for game names, companies, metrics
        game_patterns = {
            r'\bhero\s*wars?\b': 'Hero Wars mobile game',
            r'\bfortnite\b': 'Fortnite battle royale',
            r'\bdau\b': 'daily active users DAU',
            r'\brevenue\b': 'revenue monetization',
            r'\blive\s*events?\b': 'live events special events',
            r'\bstrateg(y|ies)\b': 'strategy strategies tactics'
        }
        
        for pattern, expansion in game_patterns.items():
            if re.search(pattern, enhanced):
                enhanced += f" {expansion}"
        
        return enhanced
    
    def _vector_search(self, query: str, k: int) -> List[Dict]:
        """Dense vector search only"""
        if not self.vector_store:
            logger.error("Vector store not available")
            return []
        
        try:
            logger.info(f"Vector search for: '{query}' (k={k})")
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            logger.info(f"Query embedding generated: {len(query_embedding)} dimensions")
            
            # Check if collection has any documents
            try:
                collection_count = self.vector_store.count()
                logger.info(f"Collection has {collection_count} documents")
                
                if collection_count == 0:
                    logger.warning("Collection is empty - no documents indexed")
                    return []
            except Exception as e:
                logger.warning(f"Could not check collection count: {e}")
            
            # Search
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.config.retrieval_k),
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"Vector search returned {len(results['documents'][0]) if results['documents'] else 0} results")
            
            # Convert to standard format
            documents = []
            if results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.5  # Convert distance to similarity
                    }
                    documents.append(doc)
                    logger.debug(f"Document {i}: score={doc['score']:.3f}, content_length={len(doc['content'])}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _sparse_search(self, query: str, k: int) -> List[Dict]:
        """Sparse BM25 search only"""
        return self.bm25.search(query, k)
    
    def _hybrid_search(self, query: str, k: int) -> List[Dict]:
        """Hybrid search with RRF fusion"""
        try:
            # Get results from both retrievers
            vector_results = self._vector_search(query, self.config.retrieval_k)
            sparse_results = self._sparse_search(query, self.config.retrieval_k)
            
            # Apply RRF (Reciprocal Rank Fusion)
            fused_results = self._reciprocal_rank_fusion(
                vector_results, 
                sparse_results,
                self.config.dense_weight,
                self.config.sparse_weight
            )
            
            # Apply MMR for diversity
            diverse_results = self._apply_mmr(fused_results, query, k * 2)
            
            # Re-rank if enabled
            if self.cross_encoder and len(diverse_results) > k:
                reranked_results = self._rerank_documents(query, diverse_results)
                return reranked_results[:k]
            
            return diverse_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, vector_results: List[Dict], sparse_results: List[Dict], 
                               dense_weight: float, sparse_weight: float, k: int = 60) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        
        # Create score maps
        vector_scores = {}
        sparse_scores = {}
        
        # Vector results
        for rank, doc in enumerate(vector_results):
            doc_key = self._get_doc_key(doc)
            vector_scores[doc_key] = dense_weight / (k + rank + 1)
        
        # Sparse results
        for rank, doc in enumerate(sparse_results):
            doc_key = self._get_doc_key(doc)
            sparse_scores[doc_key] = sparse_weight / (k + rank + 1)
        
        # Combine scores
        all_docs = {}
        all_doc_keys = set(vector_scores.keys()) | set(sparse_scores.keys())
        
        for doc_key in all_doc_keys:
            rrf_score = vector_scores.get(doc_key, 0) + sparse_scores.get(doc_key, 0)
            
            # Find the document object
            doc = None
            for result_list in [vector_results, sparse_results]:
                for candidate_doc in result_list:
                    if self._get_doc_key(candidate_doc) == doc_key:
                        doc = candidate_doc
                        break
                if doc:
                    break
            
            if doc:
                doc = doc.copy()
                doc["score"] = rrf_score
                all_docs[doc_key] = doc
        
        # Sort by fused score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        
        return sorted_docs
    
    def _get_doc_key(self, doc: Dict) -> str:
        """Generate unique key for document"""
        metadata = doc.get("metadata", {})
        return f"{metadata.get('doc_id', 'unknown')}_{metadata.get('chunk_id', 0)}"
    
    def _apply_mmr(self, documents: List[Dict], query: str, k: int) -> List[Dict]:
        """Apply Maximum Marginal Relevance for diversity"""
        if len(documents) <= k:
            return documents
        
        try:
            # Get query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Get document embeddings
            doc_texts = [doc["content"] for doc in documents]
            doc_embeddings = np.array(self.embeddings.embed_documents(doc_texts))
            
            # MMR algorithm
            selected = []
            remaining = list(range(len(documents)))
            
            # Select first document (highest relevance)
            first_idx = 0
            selected.append(first_idx)
            remaining.remove(first_idx)
            
            # Select remaining documents
            while len(selected) < k and remaining:
                mmr_scores = []
                
                for idx in remaining:
                    # Relevance score (cosine similarity with query)
                    relevance = np.dot(query_embedding, doc_embeddings[idx])
                    
                    # Diversity score (max similarity with selected documents)
                    if selected:
                        similarities = [np.dot(doc_embeddings[idx], doc_embeddings[sel_idx]) 
                                      for sel_idx in selected]
                        max_similarity = max(similarities)
                    else:
                        max_similarity = 0
                    
                    # MMR score
                    mmr_score = (self.config.mmr_lambda * relevance - 
                               (1 - self.config.mmr_lambda) * max_similarity)
                    mmr_scores.append((mmr_score, idx))
                
                # Select document with highest MMR score
                best_score, best_idx = max(mmr_scores, key=lambda x: x[0])
                selected.append(best_idx)
                remaining.remove(best_idx)
            
            # Return selected documents
            return [documents[idx] for idx in selected]
            
        except Exception as e:
            logger.error(f"MMR application failed: {e}")
            return documents[:k]
    
    def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Re-rank documents using cross-encoder"""
        if not self.cross_encoder:
            return documents
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc["content"]) for doc in documents]
            
            # Get scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update document scores and sort
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return documents
    
    def list_documents(self) -> Dict:
        """List all indexed documents"""
        try:
            if self.vector_store:
                collection_info = self.vector_store.get()
                return {
                    "total_chunks": len(collection_info["ids"]),
                    "collection_name": self.config.chroma_collection_name,
                    "document_store_size": len(self.document_store)
                }
            return {"error": "Vector store not available"}
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        stats = {
            "config": {
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
                "mmr_lambda": self.config.mmr_lambda,
                "use_cross_encoder": self.config.use_cross_encoder
            },
            "vector_store": "available" if self.vector_store else "unavailable",
            "bm25": "whoosh" if self.bm25.use_whoosh else "rank_bm25",
            "cross_encoder": "available" if self.cross_encoder else "unavailable",
            "embedding_cache_size": len(self.embeddings.cache)
        }
        
        # Add collection stats if available
        try:
            if self.vector_store:
                collection_info = self.vector_store.get()
                stats["total_documents"] = len(collection_info["ids"])
        except:
            pass
        
        return stats