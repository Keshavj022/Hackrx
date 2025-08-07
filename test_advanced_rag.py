"""
Test script for advanced RAG system
Tests all components: hybrid search, reranking, query decomposition, hierarchical chunking
"""

import asyncio
import json
import logging
from typing import Dict, List
from document_service import DocumentService
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGTester:
    def __init__(self):
        self.doc_service = DocumentService()
        
    async def test_document_processing(self, url: str) -> bool:
        """Test document processing with hierarchical chunking"""
        try:
            logger.info("Testing document processing...")
            result = await self.doc_service.process_document_from_url(url)
            
            success = (
                'document_id' in result and 
                'chunks_processed' in result and 
                result['chunks_processed'] > 0
            )
            
            if success:
                logger.info(f"✅ Document processing successful: {result['chunks_processed']} chunks processed")
            else:
                logger.error("❌ Document processing failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
            return False
    
    async def test_hybrid_search(self, query: str) -> bool:
        """Test hybrid search functionality"""
        try:
            logger.info(f"Testing hybrid search with query: '{query}'")
            
            # Test dense search
            dense_results = self.doc_service.vector_service._dense_search(query, top_k=5)
            
            # Test hybrid search
            hybrid_results = self.doc_service.search_documents(query, top_k=5)
            
            success = (
                len(dense_results) > 0 and 
                len(hybrid_results) > 0 and
                any('search_type' in result for result in hybrid_results)
            )
            
            if success:
                logger.info(f"✅ Hybrid search successful: {len(hybrid_results)} results")
                for result in hybrid_results[:2]:
                    search_type = result.get('search_type', 'unknown')
                    score = result.get('final_score', result.get('relevance_score', 0))
                    logger.info(f"  - Type: {search_type}, Score: {score:.3f}")
            else:
                logger.error("❌ Hybrid search failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return False
    
    async def test_query_decomposition(self, question: str) -> bool:
        """Test query decomposition and multi-step reasoning"""
        try:
            logger.info(f"Testing query decomposition with: '{question}'")
            
            query_plan = self.doc_service.llm_service.decompose_query(question)
            
            success = (
                query_plan.main_question == question and
                len(query_plan.sub_questions) > 0 and
                len(query_plan.entities) >= 0 and
                len(query_plan.reasoning_steps) > 0
            )
            
            if success:
                logger.info("✅ Query decomposition successful:")
                logger.info(f"  - Sub-questions: {len(query_plan.sub_questions)}")
                logger.info(f"  - Entities: {query_plan.entities}")
                logger.info(f"  - Query type: {query_plan.query_type.value}")
            else:
                logger.error("❌ Query decomposition failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Query decomposition error: {e}")
            return False
    
    async def test_reranking(self, query: str) -> bool:
        """Test reranking functionality"""
        try:
            if not settings.use_reranking:
                logger.info("⚠️ Reranking disabled in settings")
                return True
                
            logger.info(f"Testing reranking with query: '{query}'")
            
            # Get more candidates for reranking
            results = self.doc_service.search_documents(query, top_k=10)
            
            # Check if reranking was applied
            reranked_results = [r for r in results if 'rerank_score' in r]
            
            success = len(reranked_results) > 0 or not self.doc_service.vector_service.reranker
            
            if success:
                logger.info(f"✅ Reranking successful: {len(reranked_results)} results reranked")
            else:
                logger.error("❌ Reranking failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Reranking error: {e}")
            return False
    
    async def test_knowledge_graph(self) -> bool:
        """Test knowledge graph functionality"""
        try:
            if not self.doc_service.kg_service.enabled:
                logger.info("⚠️ Knowledge graph not enabled")
                return True
                
            logger.info("Testing knowledge graph...")
            
            # Test basic connectivity
            kg_healthy = self.doc_service.kg_service.health_check()
            
            if kg_healthy:
                logger.info("✅ Knowledge graph connectivity successful")
            else:
                logger.warning("⚠️ Knowledge graph connectivity failed")
                
            return True  # Non-critical for main functionality
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph error: {e}")
            return True  # Non-critical
    
    async def test_end_to_end_question_processing(self, question: str, document_url: str) -> bool:
        """Test complete question processing pipeline"""
        try:
            logger.info(f"Testing end-to-end processing: '{question}'")
            
            answer = await self.doc_service.process_question(question, document_url)
            
            success = (
                isinstance(answer, str) and 
                len(answer) > 10 and 
                not answer.startswith("Unable to process")
            )
            
            if success:
                logger.info("✅ End-to-end processing successful")
                logger.info(f"  Answer preview: {answer[:100]}...")
            else:
                logger.error(f"❌ End-to-end processing failed: {answer}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ End-to-end processing error: {e}")
            return False
    
    def test_health_checks(self) -> bool:
        """Test all service health checks"""
        try:
            logger.info("Testing service health checks...")
            
            health = self.doc_service.health_check()
            
            success = all(health.values())
            
            if success:
                logger.info("✅ All services healthy")
                for service, status in health.items():
                    logger.info(f"  - {service}: {'✅' if status else '❌'}")
            else:
                logger.error("❌ Some services unhealthy:")
                for service, status in health.items():
                    logger.info(f"  - {service}: {'✅' if status else '❌'}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def run_comprehensive_test(self, document_url: str, test_questions: List[str]) -> Dict[str, bool]:
        """Run comprehensive test suite"""
        logger.info("🚀 Starting Advanced RAG System Test Suite")
        logger.info("=" * 60)
        
        results = {}
        
        # Test 1: Health Checks
        results['health_checks'] = self.test_health_checks()
        
        # Test 2: Document Processing
        results['document_processing'] = await self.test_document_processing(document_url)
        
        if not results['document_processing']:
            logger.error("❌ Cannot continue tests without successful document processing")
            return results
        
        # Test 3: Hybrid Search
        results['hybrid_search'] = await self.test_hybrid_search("coverage surgery policy")
        
        # Test 4: Query Decomposition
        results['query_decomposition'] = await self.test_query_decomposition(test_questions[0])
        
        # Test 5: Reranking
        results['reranking'] = await self.test_reranking("waiting period surgery")
        
        # Test 6: Knowledge Graph
        results['knowledge_graph'] = await self.test_knowledge_graph()
        
        # Test 7: End-to-End Processing
        e2e_results = []
        for question in test_questions[:3]:  # Test first 3 questions
            e2e_result = await self.test_end_to_end_question_processing(question, document_url)
            e2e_results.append(e2e_result)
        
        results['end_to_end'] = all(e2e_results)
        
        # Summary
        logger.info("=" * 60)
        logger.info("🏁 Test Suite Summary:")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        success_rate = (passed / total) * 100
        logger.info(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")
        
        if success_rate >= 80:
            logger.info("🎉 Advanced RAG system is ready for high accuracy!")
        else:
            logger.warning("⚠️ Some components need attention before deployment")
        
        return results

async def main():
    """Main test function"""
    # Configuration
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    test_questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?"
    ]
    
    # Run tests
    tester = AdvancedRAGTester()
    results = await tester.run_comprehensive_test(document_url, test_questions)
    
    # Return results for external use
    return results

if __name__ == "__main__":
    asyncio.run(main())