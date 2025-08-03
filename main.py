"""FastAPI application for document query-retrieval system"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
from contextlib import asynccontextmanager

from config import settings, validate_settings
from document_service import DocumentService
from database import DatabaseManager, init_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

doc_processor: Optional[DocumentService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    try:
        validate_settings()
        init_database()
        
        global doc_processor
        doc_processor = DocumentService()
        
        health = doc_processor.health_check()
        if not all(health.values()):
            logger.warning("Some components are unhealthy")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    
    yield

app = FastAPI(
    title=settings.app_name,
    description="Process documents and answer questions using AI",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    system_stats: Dict[str, Any]
    health_check: Dict[str, bool]

# Middleware for performance tracking
@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    
    # Save performance metrics
    try:
        with DatabaseManager() as db:
            db.save_performance_metric(
                endpoint=request.url.path,
                response_time=processing_time,
                status_code=response.status_code,
                document_chunks=getattr(request.state, 'document_chunks', None),
                questions_count=getattr(request.state, 'questions_count', None)
            )
    except Exception as e:
        logger.error(f"Failed to save performance metrics: {e}")
    
    # Add performance header
    response.headers["X-Processing-Time"] = str(round(processing_time, 3))
    
    return response

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic system info"""
    return HealthResponse(
        status="healthy",
        message=f"{settings.app_name} is running",
        details={
            "version": settings.app_version,
            "tech_stack": {
                "backend": "FastAPI",
                "vector_db": "Pinecone",
                "llm": "GPT-4",
                "database": "PostgreSQL"
            }
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    if doc_processor is None:
        return HealthResponse(
            status="unhealthy",
            message="System not initialized",
            details={"error": "Document processor not available"}
        )
    
    try:
        health = doc_processor.health_check()
        
        if all(health.values()):
            return HealthResponse(
                status="healthy",
                message="All systems operational",
                details=health
            )
        else:
            return HealthResponse(
                status="degraded",
                message="Some components are unhealthy",
                details=health
            )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            details={"error": str(e)}
        )

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats(token: str = Depends(verify_token)):
    """Get detailed system statistics"""
    if doc_processor is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = doc_processor.get_system_stats()
        health = doc_processor.health_check()
        
        return StatsResponse(
            system_stats=stats,
            health_check=health
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_queries(
    request: QueryRequest,
    http_request: Request,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Process document and answer questions"""
    if doc_processor is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    
    try:
        http_request.state.questions_count = len(request.questions)
        
        answers = await doc_processor.process_multiple_questions(
            questions=request.questions,
            document_url=request.documents
        )
        
        processing_time = time.time() - start_time
        
        # Save query results
        try:
            client_ip = http_request.client.host
            user_agent = http_request.headers.get("user-agent", "")
            
            with DatabaseManager() as db:
                db.save_query_result(
                    document_url=request.documents,
                    questions=request.questions,
                    answers=answers,
                    processing_time=processing_time,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
        except Exception as e:
            logger.error(f"Failed to save query results: {e}")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/status")
async def get_status():
    """Get system status and configuration"""
    if doc_processor is None:
        return {"status": "unhealthy", "message": "System not initialized"}
    
    try:
        health = doc_processor.health_check()
        stats = doc_processor.get_system_stats()
        
        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "components": health,
            "tech_stack": {
                "backend": "FastAPI",
                "vector_db": "Pinecone",
                "llm": "GPT-4",
                "database": "PostgreSQL"
            },
            "performance": stats.get('performance_stats', {}),
            "vector_stats": stats.get('vector_stats', {})
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)