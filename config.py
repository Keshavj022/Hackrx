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
    bearer_token: str = "679b076ea66e474132c8ea9edcfd3fd06a608834c6ab98900d1bec673ed9fe3c"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.1
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "countzero"
    pinecone_dimension: int = 512  # For text-embedding-ada-002
    pinecone_metric: str = "cosine"
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "countzero"
    postgres_user: str = "keshav"
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
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True