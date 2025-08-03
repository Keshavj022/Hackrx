"""
Database models and connection management for PostgreSQL
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from config import settings

logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(settings.database_url, echo=settings.debug)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_hash = Column(String, nullable=False, unique=True)
    total_chunks = Column(Integer, nullable=False)
    processed_at = Column(DateTime, default=func.now())
    file_size = Column(Integer)
    file_type = Column(String)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    pinecone_id = Column(String, nullable=False, unique=True)
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    document_url = Column(String, nullable=False)
    questions = Column(JSON, nullable=False)  # List of questions
    answers = Column(JSON, nullable=False)    # List of answers
    processing_time = Column(Float)           # Processing time in seconds
    created_at = Column(DateTime, default=func.now())
    ip_address = Column(String)
    user_agent = Column(String)

class PerformanceMetrics(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, nullable=False)
    response_time = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    document_chunks = Column(Integer)
    questions_count = Column(Integer)
    timestamp = Column(DateTime, default=func.now())

# Database connection management
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

# Database operations
class DatabaseManager:
    def __init__(self):
        self.session = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
    
    def save_document(self, url: str, filename: str, content_hash: str, 
                     total_chunks: int, file_size: int = None, 
                     file_type: str = None) -> Document:
        """Save document metadata"""
        # Check if document already exists
        existing = self.session.query(Document).filter(
            Document.content_hash == content_hash
        ).first()
        
        if existing:
            logger.info(f"Document already exists: {filename}")
            return existing
        
        document = Document(
            url=url,
            filename=filename,
            content_hash=content_hash,
            total_chunks=total_chunks,
            file_size=file_size,
            file_type=file_type
        )
        
        self.session.add(document)
        self.session.flush()  # Get the ID
        logger.info(f"Saved document: {filename} with {total_chunks} chunks")
        return document
    
    def save_document_chunk(self, document_id: int, chunk_index: int, 
                           content: str, pinecone_id: str, 
                           chunk_metadata: Dict[str, Any] = None) -> DocumentChunk:
        """Save document chunk"""
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            pinecone_id=pinecone_id,
            chunk_metadata=chunk_metadata or {}
        )
        
        self.session.add(chunk)
        return chunk
    
    def save_query_result(self, document_url: str, questions: List[str], 
                         answers: List[str], processing_time: float,
                         ip_address: str = None, user_agent: str = None) -> Query:
        """Save query and results"""
        query = Query(
            document_url=document_url,
            questions=questions,
            answers=answers,
            processing_time=processing_time,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.session.add(query)
        return query
    
    def save_performance_metric(self, endpoint: str, response_time: float,
                               status_code: int, document_chunks: int = None,
                               questions_count: int = None) -> PerformanceMetrics:
        """Save performance metrics"""
        metric = PerformanceMetrics(
            endpoint=endpoint,
            response_time=response_time,
            status_code=status_code,
            document_chunks=document_chunks,
            questions_count=questions_count
        )
        
        self.session.add(metric)
        return metric
    
    def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash"""
        return self.session.query(Document).filter(
            Document.content_hash == content_hash
        ).first()
    
    def get_recent_queries(self, limit: int = 10) -> List[Query]:
        """Get recent queries"""
        return self.session.query(Query).order_by(
            Query.created_at.desc()
        ).limit(limit).all()
    
    def get_performance_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for the last N hours"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        metrics = self.session.query(PerformanceMetrics).filter(
            PerformanceMetrics.timestamp >= cutoff
        ).all()
        
        if not metrics:
            return {"total_requests": 0, "avg_response_time": 0, "success_rate": 0}
        
        total_requests = len(metrics)
        avg_response_time = sum(m.response_time for m in metrics) / total_requests
        successful_requests = len([m for m in metrics if m.status_code == 200])
        success_rate = successful_requests / total_requests * 100
        
        return {
            "total_requests": total_requests,
            "avg_response_time": round(avg_response_time, 2),
            "success_rate": round(success_rate, 2),
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests
        }