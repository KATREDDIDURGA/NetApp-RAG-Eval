"""
Configuration settings for the RAG system
Centralized configuration with environment variable loading
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Centralized configuration for the RAG system"""
    
    # =============================================================================
    # API & Authentication
    # =============================================================================
    upstage_api_key: str = field(default_factory=lambda: os.getenv("UPSTAGE_API_KEY", ""))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_key_header: str = field(default_factory=lambda: os.getenv("API_KEY_HEADER", "X-API-Key"))
    allowed_api_keys: List[str] = field(default_factory=list)
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    # LLM Settings
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "solar-1-mini-chat"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2000")))
    
    # Embedding Settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "solar-embedding-1-large-query"))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    embedding_max_retries: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_RETRIES", "3")))
    
    # =============================================================================
    # Document Processing
    # =============================================================================
    # Chunking Parameters
    max_chunk_size: int = field(default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "500")))
    min_chunk_size: int = field(default_factory=lambda: int(os.getenv("MIN_CHUNK_SIZE", "50")))
    chunk_overlap: float = field(default_factory=lambda: float(os.getenv("CHUNK_OVERLAP", "0.15")))  # 15% overlap
    
    # Text Processing
    normalize_unicode: bool = field(default_factory=lambda: os.getenv("NORMALIZE_UNICODE", "true").lower() == "true")
    remove_extra_whitespace: bool = field(default_factory=lambda: os.getenv("REMOVE_EXTRA_WHITESPACE", "true").lower() == "true")
    dehyphenate_text: bool = field(default_factory=lambda: os.getenv("DEHYPHENATE_TEXT", "true").lower() == "true")
    detect_language: bool = field(default_factory=lambda: os.getenv("DETECT_LANGUAGE", "true").lower() == "true")
    
    # HTML Processing
    extract_main_content: bool = field(default_factory=lambda: os.getenv("EXTRACT_MAIN_CONTENT", "true").lower() == "true")
    remove_navigation: bool = field(default_factory=lambda: os.getenv("REMOVE_NAVIGATION", "true").lower() == "true")
    remove_ads: bool = field(default_factory=lambda: os.getenv("REMOVE_ADS", "true").lower() == "true")
    remove_scripts: bool = field(default_factory=lambda: os.getenv("REMOVE_SCRIPTS", "true").lower() == "true")
    
    # PDF Processing
    ocr_fallback: bool = field(default_factory=lambda: os.getenv("OCR_FALLBACK", "true").lower() == "true")
    extract_tables: bool = field(default_factory=lambda: os.getenv("EXTRACT_TABLES", "true").lower() == "true")
    preserve_layout: bool = field(default_factory=lambda: os.getenv("PRESERVE_LAYOUT", "false").lower() == "true")
    
    # =============================================================================
    # Vector Store & Indexing
    # =============================================================================
    # ChromaDB Settings
    chroma_persist_directory: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vectorstore"))
    chroma_collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "documents"))
    
    # HNSW Index Parameters
    hnsw_space: str = field(default_factory=lambda: os.getenv("HNSW_SPACE", "cosine"))
    hnsw_m: int = field(default_factory=lambda: int(os.getenv("HNSW_M", "32")))
    hnsw_ef_construction: int = field(default_factory=lambda: int(os.getenv("HNSW_EF_CONSTRUCTION", "200")))
    hnsw_ef_search: int = field(default_factory=lambda: int(os.getenv("HNSW_EF_SEARCH", "128")))
    
    # BM25 Settings
    bm25_k1: float = field(default_factory=lambda: float(os.getenv("BM25_K1", "1.2")))
    bm25_b: float = field(default_factory=lambda: float(os.getenv("BM25_B", "0.75")))
    use_stemming: bool = field(default_factory=lambda: os.getenv("USE_STEMMING", "true").lower() == "true")
    
    # =============================================================================
    # Retrieval Configuration
    # =============================================================================
    # Search Parameters
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "40")))  # Initial retrieval
    final_k: int = field(default_factory=lambda: int(os.getenv("FINAL_K", "8")))  # Final results after re-ranking
    mmr_lambda: float = field(default_factory=lambda: float(os.getenv("MMR_LAMBDA", "0.5")))  # MMR diversity param
    
    # Hybrid Search Weights
    dense_weight: float = field(default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "0.6")))
    sparse_weight: float = field(default_factory=lambda: float(os.getenv("SPARSE_WEIGHT", "0.4")))
    
    # Re-ranking
    use_cross_encoder: bool = field(default_factory=lambda: os.getenv("USE_CROSS_ENCODER", "true").lower() == "true")
    cross_encoder_model: str = field(default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    
    # Query Processing
    expand_acronyms: bool = field(default_factory=lambda: os.getenv("EXPAND_ACRONYMS", "true").lower() == "true")
    spell_correction: bool = field(default_factory=lambda: os.getenv("SPELL_CORRECTION", "false").lower() == "true")
    
    # =============================================================================
    # Agent & LangGraph
    # =============================================================================
    # Quality Thresholds
    min_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6")))
    min_faithfulness_score: float = field(default_factory=lambda: float(os.getenv("MIN_FAITHFULNESS_SCORE", "0.7")))
    max_critique_issues: int = field(default_factory=lambda: int(os.getenv("MAX_CRITIQUE_ISSUES", "2")))
    
    # PII Detection
    detect_pii: bool = field(default_factory=lambda: os.getenv("DETECT_PII", "true").lower() == "true")
    pii_patterns: List[str] = field(default_factory=list)
    
    # Agent Behavior
    max_retry_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_RETRY_ATTEMPTS", "2")))
    enable_fallback: bool = field(default_factory=lambda: os.getenv("ENABLE_FALLBACK", "true").lower() == "true")
    
    # =============================================================================
    # Caching
    # =============================================================================
    # Cache Settings
    enable_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true")
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))  # 1 hour
    embedding_cache_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_CACHE_SIZE", "10000")))
    retrieval_cache_size: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_CACHE_SIZE", "1000")))
    answer_cache_size: int = field(default_factory=lambda: int(os.getenv("ANSWER_CACHE_SIZE", "500")))
    
    # =============================================================================
    # File Paths
    # =============================================================================
    # Data Directories
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    raw_data_dir: Optional[Path] = field(default=None)
    processed_data_dir: Optional[Path] = field(default=None)
    
    # Processing Files
    processed_docs_file: str = field(default="processed_docs.json")  # Match processor filename
    metadata_file: str = field(default="document_metadata.json")
    
    # =============================================================================
    # Performance & Limits
    # =============================================================================
    # API Limits
    max_query_length: int = field(default_factory=lambda: int(os.getenv("MAX_QUERY_LENGTH", "1000")))
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "50")))
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "100")))
    rate_limit_window_minutes: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW_MINUTES", "60")))
    
    # Performance Targets
    target_latency_p95_seconds: float = field(default_factory=lambda: float(os.getenv("TARGET_LATENCY_P95", "3.0")))
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")))
    
    # =============================================================================
    # Logging & Observability
    # =============================================================================
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))  # json or text
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE", None))
    
    # Metrics & Tracing
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true")
    enable_tracing: bool = field(default_factory=lambda: os.getenv("ENABLE_TRACING", "true").lower() == "true")
    trace_sample_rate: float = field(default_factory=lambda: float(os.getenv("TRACE_SAMPLE_RATE", "0.1")))
    
    # =============================================================================
    # Development & Debug
    # =============================================================================
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    save_intermediate_results: bool = field(default_factory=lambda: os.getenv("SAVE_INTERMEDIATE_RESULTS", "false").lower() == "true")
    enable_debug_endpoints: bool = field(default_factory=lambda: os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").lower() == "true")
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set up data directories
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Set up API keys list
        api_keys_str = os.getenv("ALLOWED_API_KEYS", "")
        self.allowed_api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
        
        # Set up PII patterns
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b',  # Phone number
        ]
        
        # Validate critical settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.upstage_api_key:
            raise ValueError("UPSTAGE_API_KEY is required but not set")
        
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("MAX_CHUNK_SIZE must be greater than MIN_CHUNK_SIZE")
        
        if not (0 <= self.chunk_overlap < 1):
            raise ValueError("CHUNK_OVERLAP must be between 0 and 1")
        
        if not (0 <= self.mmr_lambda <= 1):
            raise ValueError("MMR_LAMBDA must be between 0 and 1")
        
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("DENSE_WEIGHT + SPARSE_WEIGHT must be positive")
        
        if self.retrieval_k < self.final_k:
            raise ValueError("RETRIEVAL_K must be >= FINAL_K")
    
    def get_acronym_expansions(self) -> Dict[str, str]:
        """Get acronym expansion dictionary"""
        return {
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "nlp": "natural language processing",
            "api": "application programming interface",
            "ui": "user interface",
            "ux": "user experience",
            "db": "database",
            "sql": "structured query language",
            "http": "hypertext transfer protocol",
            "url": "uniform resource locator",
            "json": "javascript object notation",
            "xml": "extensible markup language",
            "css": "cascading style sheets",
            "html": "hypertext markup language",
            "pdf": "portable document format",
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "gpt": "generative pre-trained transformer",
        }
    
    def get_summary(self) -> Dict:
        """Get configuration summary for logging"""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "final_k": self.final_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "enable_caching": self.enable_caching,
            "debug_mode": self.debug_mode,
        }

# Global config instance
config = Config()

# Ensure post-init is called
config.__post_init__()