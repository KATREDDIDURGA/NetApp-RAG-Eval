"""
FastAPI server for RAG system
Provides REST API endpoints for document ingestion and querying
"""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from cachetools import TTLCache
import logging

from config import config
from processor import DocumentProcessor
from retriever import HybridRetriever
from agent import RAGAgent

# =============================================================================
# Setup Logging
# =============================================================================
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State & Caches
# =============================================================================
app_state = {
    "processor": None,
    "retriever": None,
    "agent": None,
    "startup_time": None,
    "request_count": 0,
    "traces": {}
}

# Initialize caches
embedding_cache = TTLCache(maxsize=config.embedding_cache_size, ttl=config.cache_ttl_seconds)
retrieval_cache = TTLCache(maxsize=config.retrieval_cache_size, ttl=config.cache_ttl_seconds)
answer_cache = TTLCache(maxsize=config.answer_cache_size, ttl=config.cache_ttl_seconds)

# Rate limiting
rate_limit_cache = TTLCache(maxsize=10000, ttl=config.rate_limit_window_minutes * 60)

# =============================================================================
# Pydantic Models
# =============================================================================
class IngestionRequest(BaseModel):
    force_reprocess: bool = Field(default=False)

class IngestionResponse(BaseModel):
    status: str
    processed_count: int
    total_count: int
    processing_time_seconds: float
    message: str
    document_ids: List[str]

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=config.max_query_length)
    mode: Optional[str] = Field(default="auto")
    k: Optional[int] = Field(default=None, ge=1, le=20)
    include_debug: bool = Field(default=False)

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    retrieval_mode: str
    trace_id: str
    timestamp: str
    processing_time_seconds: float
    debug_info: Optional[Dict] = None

class TraceResponse(BaseModel):
    trace_id: str
    query: str
    response: QueryResponse
    timestamp: str
    processing_steps: List[Dict]

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    cache_stats: Dict
    system_info: Dict

# =============================================================================
# Lifecycle Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG system...")
    app_state["startup_time"] = time.time()
    
    try:
        logger.info("Initializing document processor...")
        # Fixed: Pass processed_data_dir directly as Path, and ensure filename matches
        app_state["processor"] = DocumentProcessor(
            config.processed_data_dir, 
            processed_docs_file=config.processed_docs_file
        )

        logger.info("Initializing retriever...")
        app_state["retriever"] = HybridRetriever(config)

        logger.info("Initializing RAG agent...")
        app_state["agent"] = RAGAgent(config, app_state["retriever"])

        logger.info("✅ RAG system startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    logger.info("Shutting down RAG system...")

# =============================================================================
# Security
# =============================================================================
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not config.allowed_api_keys:
        return True
    if not credentials or credentials.credentials not in config.allowed_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# =============================================================================
# Utility
# =============================================================================
def generate_cache_key(data: Any) -> str:
    content = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def get_cache_stats() -> Dict:
    return {
        "embedding_cache": {"size": len(embedding_cache), "max_size": embedding_cache.maxsize},
        "retrieval_cache": {"size": len(retrieval_cache), "max_size": retrieval_cache.maxsize},
        "answer_cache": {"size": len(answer_cache), "max_size": answer_cache.maxsize}
    }

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation system with hybrid search",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    # Get additional system info
    system_info = {
        "config_summary": config.get_summary(),
        "components_initialized": all([app_state["processor"], app_state["retriever"], app_state["agent"]])
    }
    
    # Add document processing stats if available
    if app_state["processor"]:
        try:
            system_info["processing_stats"] = app_state["processor"].get_processing_stats()
        except Exception as e:
            logger.warning(f"Could not get processing stats: {e}")
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        total_requests=app_state["request_count"],
        cache_stats=get_cache_stats(),
        system_info=system_info
    )

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    app_state["request_count"] += 1
    start_time = time.time()
    
    if not app_state["processor"]:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        logger.info(f"📂 Starting ingestion (force_reprocess={request.force_reprocess})")
        
        # Fixed: Removed unsupported batch_size argument
        result = await asyncio.to_thread(
            app_state["processor"].process_directory,
            config.raw_data_dir,
            force_reprocess=request.force_reprocess
        )

        # Index documents if any were processed
        if result["processed_count"] > 0:
            logger.info(f"📦 Indexing {len(result['document_ids'])} documents...")
            try:
                # Run indexing in background thread to prevent blocking
                await asyncio.to_thread(
                    app_state["retriever"].index_documents, 
                    result["document_ids"]
                )
                logger.info("✅ Indexing completed successfully")
            except Exception as index_error:
                logger.error(f"❌ Indexing failed: {index_error}")
                # Schedule background retry
                background_tasks.add_task(index_processed_documents, result["document_ids"])
        else:
            logger.info("No new documents to index")
        
        processing_time = time.time() - start_time
        
        return IngestionResponse(
            status="success",
            processed_count=result["processed_count"],
            total_count=result["total_count"],
            processing_time_seconds=processing_time,
            message=f"Successfully processed {result['processed_count']} documents and indexed {len(result['document_ids'])} total documents",
            document_ids=result["document_ids"]
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, _: bool = Depends(verify_api_key)):
    app_state["request_count"] += 1
    start_time = time.time()
    
    if not app_state["agent"]:
        raise HTTPException(status_code=503, detail="RAG agent not initialized")
    
    try:
        trace_id = f"trace_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Run query in background thread
        result = await asyncio.to_thread(
            app_state["agent"].query, 
            request.query, 
            trace_id=trace_id
        )
        
        processing_time = time.time() - start_time
        
        response_data = {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "retrieval_mode": result["retrieval_mode"],
            "trace_id": trace_id,
            "timestamp": result["timestamp"],
            "processing_time_seconds": processing_time
        }
        
        # Add debug info if requested
        if request.include_debug:
            response_data["debug_info"] = {
                "document_count": len(result.get("documents", [])),
                "critique": result.get("critique", {}),
                "trace_id": trace_id
            }
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/documents")
async def list_documents(_: bool = Depends(verify_api_key)):
    """List all processed documents"""
    if not app_state["retriever"]:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        document_info = app_state["retriever"].list_documents()
        return document_info
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/stats")
async def get_system_stats(_: bool = Depends(verify_api_key)):
    """Get system statistics"""
    stats = {
        "uptime_seconds": time.time() - app_state["startup_time"] if app_state["startup_time"] else 0,
        "total_requests": app_state["request_count"],
        "cache_stats": get_cache_stats()
    }
    
    if app_state["retriever"]:
        try:
            stats["retriever_stats"] = app_state["retriever"].get_stats()
        except Exception as e:
            logger.warning(f"Could not get retriever stats: {e}")
    
    if app_state["processor"]:
        try:
            stats["processing_stats"] = app_state["processor"].get_processing_stats()
        except Exception as e:
            logger.warning(f"Could not get processing stats: {e}")
    
    return stats

@app.delete("/cache")
async def clear_caches(_: bool = Depends(verify_api_key)):
    """Clear all caches"""
    embedding_cache.clear()
    retrieval_cache.clear()
    answer_cache.clear()
    
    return {"message": "All caches cleared successfully"}

async def index_processed_documents(document_ids: List[str]):
    """Background task to index processed documents"""
    try:
        logger.info(f"🔄 Background indexing {len(document_ids)} documents...")
        await asyncio.to_thread(app_state["retriever"].index_documents, document_ids)
        logger.info("✅ Background indexing completed")
    except Exception as e:
        logger.error(f"❌ Background indexing failed: {e}")

# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": f"File not found: {str(exc)}"}
    )

# =============================================================================
# Server Runner
# =============================================================================
def run_server():
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug_mode,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    run_server()