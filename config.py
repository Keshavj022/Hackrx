"""
Configuration settings for the LLM-Powered Intelligent Query-Retrieval System
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Authentication
    bearer_token: Optional[str] = None
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.1
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"
    
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
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 5
    
    # Performance Settings
    max_concurrent_requests: int = 10
    request_timeout: int = 120
    
    # Testing Configuration
    ngrok_url: Optional[str] = None
    api_base_url: str = "http://localhost:8000"
    
    # File Server Configuration
    file_server_dir: str = "data"
    file_server_port: int = 8001
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validation
def validate_settings():
    """Validate that all required settings are provided"""
    errors = []
    
    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    
    if not settings.pinecone_api_key:
        errors.append("PINECONE_API_KEY is required")
    
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is required")
    
    if not settings.postgres_host:
        errors.append("POSTGRES_HOST is required")
    
    if not settings.postgres_db:
        errors.append("POSTGRES_DB is required")
    
    if not settings.postgres_user:
        errors.append("POSTGRES_USER is required")
    
    if not settings.pinecone_environment:
        errors.append("PINECONE_ENVIRONMENT is required")
    
    if not settings.pinecone_index_name:
        errors.append("PINECONE_INDEX_NAME is required")
    
    if not settings.bearer_token:
        errors.append("BEARER_TOKEN is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True