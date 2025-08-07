"""
Configuration settings for the LLM-Powered Intelligent Query-Retrieval System
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Authentication
    bearer_token: Optional[str] = None
    
    # FAISS Configuration (replaces Pinecone)
    vector_dimension: int = 1536  # text-embedding-3-small dimension
    faiss_index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity
    
    # PostgreSQL Configuration
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Database URL
    @property
    def database_url(self) -> str:
        if self.postgres_password:
            from urllib.parse import quote_plus
            encoded_password = quote_plus(self.postgres_password)
            return f"postgresql://{self.postgres_user}:{encoded_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        return f"postgresql://{self.postgres_user}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Document Processing - Optimized for performance
    chunk_size: int = 1024  # Increased for better context
    chunk_overlap: int = 200  # Increased overlap for better continuity
    max_retrieval_docs: int = 15  # Increased for better coverage
    parent_chunk_size: int = 4096  # Larger parent chunks
    child_chunk_size: int = 1024  # Larger child chunks
    
    # Advanced RAG Settings - Performance optimized
    use_hybrid_search: bool = True
    use_reranking: bool = True
    use_query_decomposition: bool = True
    use_semantic_chunking: bool = True
    use_contextual_retrieval: bool = True
    
    # Search and Ranking Parameters
    rerank_top_k: int = 30  # More candidates for reranking
    final_top_k: int = 12   # More final results for better context
    dense_weight: float = 0.6  # Weight for dense search
    sparse_weight: float = 0.4  # Weight for sparse search
    
    # Neo4j Configuration (optional)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Cross-encoder model for reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Performance Settings - Optimized for speed
    max_concurrent_requests: int = 20
    request_timeout: int = 60  # Reduced timeout
    embedding_batch_size: int = 50  # Batch embeddings for efficiency
    cache_embeddings: bool = True
    
    # Async Processing
    use_async_processing: bool = True
    max_workers: int = 4
    
    # Testing Configuration
    ngrok_url: Optional[str] = None
    api_base_url: str = "http://localhost:8000"
    
    # File Server Configuration
    file_server_dir: str = "data"
    file_server_port: int = 8001
    
    # Backward compatibility properties
    @property
    def pinecone_dimension(self) -> int:
        """Backward compatibility for existing code"""
        return self.vector_dimension
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Updated validation function
def validate_settings():
    """Validate that all required settings are provided"""
    errors = []
    
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is required")
    
    if not settings.postgres_host:
        errors.append("POSTGRES_HOST is required")
    
    if not settings.postgres_db:
        errors.append("POSTGRES_DB is required")
    
    if not settings.postgres_user:
        errors.append("POSTGRES_USER is required")
    
    if not settings.bearer_token:
        errors.append("BEARER_TOKEN is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True