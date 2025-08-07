"""Advanced RAG System with Sentence-Window, Auto-Merging, and RAG Triad Evaluation"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from config import settings, validate_settings
from document_service import DocumentService
from vector_service import VectorService
from llm_service import LLMService
from knowledge_graph_service import KnowledgeGraphService
from database import DatabaseManager, init_database

import sys

# Configure logging to write to stderr instead of stdout to avoid mixing with JSON responses
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # This ensures logs go to stderr, not stdout
)
logger = logging.getLogger(__name__)

# Global services with health monitoring
doc_processor: Optional[DocumentService] = None
vector_service: Optional[VectorService] = None
llm_service: Optional[LLMService] = None
knowledge_graph: Optional[KnowledgeGraphService] = None

# Token usage tracking
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.request_count = 0
        self.average_tokens_per_request = 0
    
    def add_request(self, tokens_used: int):
        self.total_tokens += tokens_used
        self.request_count += 1
        self.average_tokens_per_request = self.total_tokens / self.request_count if self.request_count > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_tokens_used': self.total_tokens,
            'total_requests': self.request_count,
            'average_tokens_per_request': round(self.average_tokens_per_request, 2)
        }

token_tracker = TokenTracker()

@dataclass
class RetrievalMetrics:
    context_relevance: float
    groundedness: float
    answer_relevance: float
    processing_time: float
    tokens_used: int
    confidence_score: float

class AdvancedRAGSystem:
    def __init__(self):
        self.document_service = None
        self.vector_service = None
        self.llm_service = None
        self.knowledge_graph = None
        
    async def initialize(self):
        global doc_processor, vector_service, llm_service, knowledge_graph
        
        # Initialize all services
        self.document_service = DocumentService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.knowledge_graph = KnowledgeGraphService()
        
        # Initialize services concurrently
        await asyncio.gather(
            self.vector_service.initialize(),
            self.llm_service.initialize(),
            self.knowledge_graph.initialize()
        )
        
        # Set global references
        doc_processor = self.document_service
        vector_service = self.vector_service
        llm_service = self.llm_service
        knowledge_graph = self.knowledge_graph
        
        logger.info("Advanced RAG System initialized with all components")
    
    async def process_query_with_advanced_rag(self, documents_url: str, questions: List[str], include_details: bool = False) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Step 1: Advanced Document Processing with pymupdf4llm
            logger.info(f"Processing document with pymupdf4llm: {documents_url}")
            document_content = await self.document_service.extract_with_pymupdf4llm(documents_url)
            
            # Step 2: Create hierarchical embeddings with sentence-window and auto-merging
            logger.info("Creating hierarchical embeddings with advanced retrieval techniques")
            await self.vector_service.process_document_with_advanced_chunking(
                document_content, 
                documents_url,
                use_sentence_window=True,
                use_auto_merging=True
            )
            
            # Step 3: Build knowledge graph for entity relationships
            await self.knowledge_graph.build_from_document(document_content, documents_url)
            
            # Step 4: Process each question with advanced RAG pipeline
            answers = []
            detailed_answers = []
            total_tokens = 0
            
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:100]}...")
                
                # Multi-stage advanced retrieval
                contexts = await self._advanced_retrieval_pipeline(question)
                
                # Generate answer with explainable reasoning using Ollama
                answer_result = await self.llm_service.generate_answer_with_advanced_reasoning(
                    question, contexts, model="llama3.1:8b"
                )
                
                # Evaluate RAG triad metrics
                metrics = await self._evaluate_rag_triad(question, contexts, answer_result['answer'])
                
                # Extract source references from contexts
                source_references = []
                for ctx in contexts[:3]:  # Top 3 contexts
                    if hasattr(ctx, 'chunk') and hasattr(ctx.chunk, 'metadata'):
                        section = ctx.chunk.metadata.get('section', 'Unknown Section')
                        source_references.append(f"Section: {section}")
                    elif isinstance(ctx, dict) and 'metadata' in ctx:
                        section = ctx['metadata'].get('section', 'Unknown Section')
                        source_references.append(f"Section: {section}")
                
                # Add basic answer to list
                answers.append(answer_result['answer'])
                total_tokens += answer_result.get('tokens_generated', 0)
                
                # Add detailed answer if requested
                if include_details:
                    detailed_answer = {
                        'answer': answer_result['answer'],
                        'confidence_score': answer_result.get('confidence', metrics.confidence_score),
                        'reasoning': answer_result.get('reasoning', 'Generated using advanced RAG pipeline with multi-stage retrieval'),
                        'source_references': source_references,
                        'limitations': answer_result.get('limitations', None),
                        'context_relevance': metrics.context_relevance,
                        'groundedness': metrics.groundedness,
                        'answer_relevance': metrics.answer_relevance
                    }
                    detailed_answers.append(detailed_answer)
                
                logger.info(f"Question {i+1} processed - Relevance: {metrics.context_relevance:.3f}, "
                          f"Groundedness: {metrics.groundedness:.3f}, "
                          f"Answer Relevance: {metrics.answer_relevance:.3f}, "
                          f"Confidence: {metrics.confidence_score:.3f}")
            
            processing_time = time.time() - start_time
            logger.info(f"Total advanced RAG processing time: {processing_time:.2f}s")
            
            # Return structured result
            result = {
                'answers': answers,
                'processing_time': processing_time,
                'total_tokens': total_tokens,
                'questions_processed': len(questions)
            }
            
            if include_details:
                result['detailed_answers'] = detailed_answers
                result['processing_metadata'] = {
                    'document_url': documents_url,
                    'processing_time_seconds': processing_time,
                    'total_tokens_used': total_tokens,
                    'questions_count': len(questions),
                    'average_confidence': sum(d['confidence_score'] for d in detailed_answers) / len(detailed_answers) if detailed_answers else 0,
                    'average_context_relevance': sum(d['context_relevance'] for d in detailed_answers) / len(detailed_answers) if detailed_answers else 0,
                    'average_groundedness': sum(d['groundedness'] for d in detailed_answers) / len(detailed_answers) if detailed_answers else 0,
                    'average_answer_relevance': sum(d['answer_relevance'] for d in detailed_answers) / len(detailed_answers) if detailed_answers else 0,
                    'tech_stack_used': {
                        'pdf_extraction': 'pymupdf4llm',
                        'vector_search': 'FAISS with HNSW',
                        'llm_model': 'llama3.1:8b (Ollama)',
                        'retrieval_methods': ['sentence-window', 'auto-merging', 'hybrid-rrf', 'cross-encoder-reranking'],
                        'evaluation_framework': 'RAG Triad (Context Relevance, Groundedness, Answer Relevance)'
                    }
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced RAG processing: {str(e)}")
            # Return graceful error response instead of 500 when possible
            if "timeout" in str(e).lower():
                raise HTTPException(status_code=408, detail="Request timeout - document processing took too long")
            elif "memory" in str(e).lower():
                raise HTTPException(status_code=413, detail="Document too large - exceeds processing limits")
            elif "connection" in str(e).lower() or "network" in str(e).lower():
                raise HTTPException(status_code=503, detail="Service temporarily unavailable - please try again")
            else:
                raise HTTPException(status_code=500, detail="Internal processing error - please contact support")
    
    async def _advanced_retrieval_pipeline(self, query: str, k: int = 10) -> List[Dict]:
        # Stage 1: Semantic search with FAISS (dense retrieval)
        dense_results = await self.vector_service.advanced_similarity_search(query, k=k)
        
        # Stage 2: BM25 sparse retrieval
        sparse_results = await self.vector_service.bm25_search(query, k=k)
        
        # Stage 3: Hybrid fusion with RRF (Reciprocal Rank Fusion)
        hybrid_results = await self.vector_service.hybrid_search_with_rrf(
            query, dense_results, sparse_results, k=k
        )
        
        # Stage 4: Knowledge graph enhancement
        kg_enhanced = await self.knowledge_graph.enhance_retrieval_with_entities(
            query, hybrid_results
        )
        
        # Stage 5: Sentence-window retrieval
        windowed_contexts = await self.vector_service.apply_sentence_window_retrieval(
            kg_enhanced, window_size=3
        )
        
        # Stage 6: Auto-merging retrieval
        merged_contexts = await self.vector_service.apply_auto_merging_retrieval(
            windowed_contexts, similarity_threshold=0.8
        )
        
        # Stage 7: Re-ranking with cross-encoder
        reranked_contexts = await self.vector_service.rerank_contexts(query, merged_contexts)
        
        return reranked_contexts[:k]
    
    async def _evaluate_rag_triad(self, question: str, contexts: List[Dict], answer: str) -> RetrievalMetrics:
        try:
            # Evaluate RAG Triad metrics concurrently
            context_relevance_task = self.llm_service.evaluate_context_relevance(question, contexts)
            groundedness_task = self.llm_service.evaluate_groundedness(contexts, answer)
            answer_relevance_task = self.llm_service.evaluate_answer_relevance(question, answer)
            
            context_relevance, groundedness, answer_relevance = await asyncio.gather(
                context_relevance_task, groundedness_task, answer_relevance_task
            )
            
            # Calculate confidence score (weighted average)
            confidence_score = (
                0.3 * context_relevance + 
                0.4 * groundedness + 
                0.3 * answer_relevance
            )
            
            return RetrievalMetrics(
                context_relevance=context_relevance,
                groundedness=groundedness,
                answer_relevance=answer_relevance,
                processing_time=0.0,  # Will be set by caller
                tokens_used=0,  # Will be tracked by LLM service
                confidence_score=confidence_score
            )
        except Exception as e:
            logger.error(f"Error evaluating RAG triad: {str(e)}")
            # Return default metrics if evaluation fails
            return RetrievalMetrics(
                context_relevance=0.5,
                groundedness=0.5,
                answer_relevance=0.5,
                processing_time=0.0,
                tokens_used=0,
                confidence_score=0.5
            )

# Global RAG system instance
rag_system: Optional[AdvancedRAGSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown with advanced RAG system"""
    try:
        validate_settings()
        init_database()
        
        global rag_system
        rag_system = AdvancedRAGSystem()
        await rag_system.initialize()
        
        logger.info("Advanced RAG System startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize advanced RAG system: {e}")
        raise
    
    yield
    
    # Cleanup resources
    logger.info("Shutting down Advanced RAG System")
    try:
        if rag_system and rag_system.document_service:
            rag_system.document_service.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")

app = FastAPI(
    title="Advanced RAG System with RAG Triad Evaluation",
    description="Intelligent Query-Retrieval System with Sentence-Window, Auto-Merging, Knowledge Graph, and RAG Triad Evaluation",
    version="2.0.0",
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

class AnswerDetails(BaseModel):
    """Detailed answer with explainable reasoning"""
    answer: str = Field(..., description="The main answer to the question")
    confidence_score: float = Field(..., description="Confidence level (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation of how the answer was derived")
    source_references: List[str] = Field(default=[], description="References to source document sections")
    limitations: Optional[str] = Field(None, description="Any limitations or caveats in the answer")
    context_relevance: float = Field(..., description="How relevant the retrieved context was (0.0 to 1.0)")
    groundedness: float = Field(..., description="How well the answer is grounded in the source (0.0 to 1.0)")
    answer_relevance: float = Field(..., description="How well the answer addresses the question (0.0 to 1.0)")

class QueryResponse(BaseModel):
    """Enhanced response with detailed explanations"""
    answers: List[str] = Field(..., description="List of answers to the questions")
    detailed_answers: Optional[List[AnswerDetails]] = Field(None, description="Detailed answers with explanations")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing statistics and metadata")

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
    """Root endpoint with advanced system info"""
    return HealthResponse(
        status="healthy",
        message=f"{settings.app_name} v2.0 - Advanced RAG System",
        details={
            "version": "2.0.0",
            "tech_stack": {
                "backend": "FastAPI",
                "vector_db": "FAISS",
                "llm": "Ollama (llama3.1:8b)",
                "pdf_extraction": "pymupdf4llm",
                "database": "SQLite/PostgreSQL",
                "knowledge_graph": "NetworkX"
            },
            "advanced_features": {
                "sentence_window_retrieval": True,
                "auto_merging_retrieval": True,
                "hybrid_search": True,
                "knowledge_graph_enhancement": True,
                "rag_triad_evaluation": True,
                "reciprocal_rank_fusion": True,
                "cross_encoder_reranking": True,
                "explainable_reasoning": True
            }
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for advanced system"""
    if rag_system is None or doc_processor is None:
        return HealthResponse(
            status="unhealthy",
            message="Advanced RAG system not initialized",
            details={"error": "System components not available"}
        )
    
    try:
        health = doc_processor.health_check()
        
        # Check vector service
        if vector_service:
            health["vector_service"] = vector_service.is_healthy()
        
        # Check LLM service  
        if llm_service:
            health["llm_service"] = llm_service.is_healthy()
            
        # Check knowledge graph
        if knowledge_graph:
            health["knowledge_graph"] = knowledge_graph.is_healthy()
        
        if all(health.values()):
            return HealthResponse(
                status="healthy",
                message="All advanced RAG components operational",
                details=health
            )
        else:
            return HealthResponse(
                status="degraded",
                message="Some advanced components are unhealthy",
                details=health
            )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Advanced health check failed: {str(e)}",
            details={"error": str(e)}
        )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_advanced_queries(
    request: QueryRequest,
    http_request: Request,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Process document and answer questions with advanced RAG pipeline"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="Advanced RAG system not initialized")
    
    start_time = time.time()
    
    try:
        http_request.state.questions_count = len(request.questions)
        
        # Use advanced RAG processing
        result = await rag_system.process_query_with_advanced_rag(
            request.documents, request.questions, include_details=False
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Advanced RAG processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        # Save query results
        try:
            client_ip = http_request.client.host
            user_agent = http_request.headers.get("user-agent", "")
            
            with DatabaseManager() as db:
                db.save_query_result(
                    document_url=request.documents,
                    questions=request.questions,
                    answers=result['answers'],
                    processing_time=processing_time,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
        except Exception as e:
            logger.error(f"Failed to save query results: {e}")
        
        return QueryResponse(answers=result['answers'])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in advanced RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/hackrx/run/detailed", response_model=QueryResponse)
async def run_advanced_queries_detailed(
    request: QueryRequest,
    http_request: Request,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Process document and answer questions with detailed explanations and RAG triad metrics"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="Advanced RAG system not initialized")
    
    start_time = time.time()
    
    try:
        http_request.state.questions_count = len(request.questions)
        
        # Use advanced RAG processing with detailed explanations
        result = await rag_system.process_query_with_advanced_rag(
            request.documents, request.questions, include_details=True
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Advanced RAG processed {len(request.questions)} questions with details in {processing_time:.2f}s")
        
        # Save query results
        try:
            client_ip = http_request.client.host
            user_agent = http_request.headers.get("user-agent", "")
            
            with DatabaseManager() as db:
                db.save_query_result(
                    document_url=request.documents,
                    questions=request.questions,
                    answers=result['answers'],
                    processing_time=processing_time,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
        except Exception as e:
            logger.error(f"Failed to save query results: {e}")
        
        # Convert detailed answers to proper format
        detailed_answers_formatted = []
        if 'detailed_answers' in result:
            for detail in result['detailed_answers']:
                formatted_detail = AnswerDetails(
                    answer=detail['answer'],
                    confidence_score=detail['confidence_score'],
                    reasoning=detail['reasoning'],
                    source_references=detail['source_references'],
                    limitations=detail.get('limitations'),
                    context_relevance=detail['context_relevance'],
                    groundedness=detail['groundedness'],
                    answer_relevance=detail['answer_relevance']
                )
                detailed_answers_formatted.append(formatted_detail)
        
        return QueryResponse(
            answers=result['answers'],
            detailed_answers=detailed_answers_formatted,
            processing_metadata=result.get('processing_metadata')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in detailed advanced RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/status")
async def get_advanced_status():
    """Get advanced system status and configuration"""
    if rag_system is None or doc_processor is None:
        return {"status": "unhealthy", "message": "Advanced RAG system not initialized"}
    
    try:
        health = doc_processor.health_check()
        stats = doc_processor.get_system_stats()
        
        # Add advanced component health
        if vector_service:
            health["vector_service"] = vector_service.is_healthy()
        if llm_service:
            health["llm_service"] = llm_service.is_healthy()
        if knowledge_graph:
            health["knowledge_graph"] = knowledge_graph.is_healthy()
        
        return {
            "status": "healthy" if all(health.values()) else "degraded",
            "components": health,
            "tech_stack": {
                "backend": "FastAPI",
                "vector_db": "FAISS",
                "llm": "Ollama (llama3.1:8b)",
                "pdf_extraction": "pymupdf4llm",
                "database": "SQLite/PostgreSQL",
                "knowledge_graph": "NetworkX"
            },
            "advanced_features": {
                "sentence_window_retrieval": True,
                "auto_merging_retrieval": True,
                "hybrid_search_rrf": True,
                "knowledge_graph_enhancement": True,
                "rag_triad_evaluation": True,
                "cross_encoder_reranking": True,
                "explainable_reasoning": True,
                "async_processing": settings.use_async_processing,
                "contextual_retrieval": settings.use_contextual_retrieval
            },
            "performance": stats.get('performance_stats', {}),
            "vector_stats": stats.get('vector_stats', {}),
            "rag_metrics": {
                "context_relevance_threshold": 0.7,
                "groundedness_threshold": 0.8,
                "answer_relevance_threshold": 0.7,
                "confidence_threshold": 0.75
            }
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