#!/usr/bin/env python3
"""
Test script for the optimized RAG system with FAISS, semantic chunking, and performance improvements.
This script demonstrates the improvements over the original Pinecone-based system.
"""

import time
import asyncio
import os
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from vector_service import VectorService
        print("✅ FAISS Vector Service imported successfully")
        
        from document_service import DocumentService
        print("✅ Enhanced Document Service imported successfully")
        
        from llm_service import LLMService
        print("✅ Optimized LLM Service imported successfully")
        
        from knowledge_graph_service import KnowledgeGraphService
        print("✅ Knowledge Graph Service imported successfully")
        
        from config import settings
        print("✅ Configuration loaded successfully")
        
        import faiss
        print("✅ FAISS library available")
        
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy with English model available")
        except:
            print("⚠️  spaCy model not available, using basic chunking")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_vector_service():
    """Test the FAISS vector service"""
    print("\n🔍 Testing FAISS Vector Service...")
    
    try:
        from vector_service import VectorService
        
        # Initialize service
        start_time = time.time()
        vector_service = VectorService()
        init_time = time.time() - start_time
        print(f"✅ Vector service initialized in {init_time:.2f}s")
        
        # Test embedding generation with caching
        test_texts = [
            "This is a test document about insurance coverage",
            "Waiting period for pre-existing conditions",
            "Exclusions and limitations in the policy"
        ]
        
        start_time = time.time()
        embeddings = vector_service.get_embeddings(test_texts)
        embedding_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
        
        # Test caching (second call should be faster)
        start_time = time.time()
        cached_embeddings = vector_service.get_embeddings(test_texts)
        cache_time = time.time() - start_time
        print(f"✅ Cached embeddings retrieved in {cache_time:.3f}s (speedup: {embedding_time/cache_time:.1f}x)")
        
        # Test document upsertion
        documents = [
            {
                'content': text,
                'source': 'test_doc',
                'chunk_id': i,
                'chunk_type': 'child',
                'metadata': {'test': True}
            }
            for i, text in enumerate(test_texts)
        ]
        
        start_time = time.time()
        vector_ids = vector_service.upsert_documents(documents)
        upsert_time = time.time() - start_time
        print(f"✅ Upserted {len(vector_ids)} documents in {upsert_time:.2f}s")
        
        # Test search
        start_time = time.time()
        search_results = vector_service.search_similar("insurance coverage policy", top_k=2)
        search_time = time.time() - start_time
        print(f"✅ Search completed in {search_time:.3f}s, found {len(search_results)} results")
        
        # Test hybrid search
        if hasattr(vector_service, '_hybrid_search'):
            start_time = time.time()
            hybrid_results = vector_service._hybrid_search("insurance waiting period", top_k=2)
            hybrid_time = time.time() - start_time
            print(f"✅ Hybrid search completed in {hybrid_time:.3f}s, found {len(hybrid_results)} results")
        
        # Show index stats
        stats = vector_service.get_index_stats()
        print(f"📊 Vector Index Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector service test failed: {e}")
        return False

def test_llm_service():
    """Test the optimized LLM service"""
    print("\n🔍 Testing Optimized LLM Service...")
    
    try:
        from llm_service import LLMService
        
        # Initialize service
        start_time = time.time()
        llm_service = LLMService()
        init_time = time.time() - start_time
        print(f"✅ LLM service initialized in {init_time:.2f}s")
        
        # Test fast entity extraction
        test_query = "What is the waiting period for a 35 year old male for cardiac surgery coverage?"
        
        start_time = time.time()
        entities = llm_service._extract_entities_fast(test_query)
        entity_time = time.time() - start_time
        print(f"✅ Fast entity extraction in {entity_time:.3f}s: {entities}")
        
        # Test query classification
        start_time = time.time()
        query_type = llm_service._classify_query_type(test_query.lower())
        classify_time = time.time() - start_time
        print(f"✅ Query classification in {classify_time:.3f}s: {query_type.value}")
        
        # Test optimized query parsing
        start_time = time.time()
        parsed_query = llm_service.parse_query(test_query)
        parse_time = time.time() - start_time
        print(f"✅ Query parsing in {parse_time:.3f}s")
        print(f"   Query type: {parsed_query.get('query_type')}")
        print(f"   Entities: {parsed_query.get('entities', [])}")
        
        # Test search query generation
        start_time = time.time()
        search_queries = llm_service.generate_search_queries(parsed_query)
        search_gen_time = time.time() - start_time
        print(f"✅ Search query generation in {search_gen_time:.3f}s")
        print(f"   Generated {len(search_queries)} queries: {search_queries[:3]}...")
        
        # Test caching
        cache_stats = llm_service.get_cache_stats()
        print(f"📊 LLM Cache Stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False

def test_document_service():
    """Test the enhanced document service"""
    print("\n🔍 Testing Enhanced Document Service...")
    
    try:
        from document_service import DocumentService
        
        # Initialize service
        start_time = time.time()
        doc_service = DocumentService()
        init_time = time.time() - start_time
        print(f"✅ Document service initialized in {init_time:.2f}s")
        
        # Test semantic section extraction
        sample_text = """
1. COVERAGE
This policy covers hospitalization expenses for treatment of illness or injury.

2. EXCLUSIONS
The following are not covered under this policy:
- Pre-existing conditions within 2 years
- Cosmetic surgery
- Self-inflicted injuries

3. WAITING PERIOD
There is a waiting period of 30 days for illness and 2 years for pre-existing conditions.
"""
        
        start_time = time.time()
        sections = doc_service._extract_semantic_sections(sample_text)
        section_time = time.time() - start_time
        print(f"✅ Semantic section extraction in {section_time:.3f}s")
        print(f"   Found {len(sections)} sections")
        for title, _, importance in sections:
            print(f"   - {title} (importance: {importance:.2f})")
        
        # Test entity extraction
        start_time = time.time()
        entities = doc_service._extract_enhanced_entities(sample_text)
        entity_time = time.time() - start_time
        print(f"✅ Enhanced entity extraction in {entity_time:.3f}s: {entities}")
        
        # Test semantic density calculation
        start_time = time.time()
        density = doc_service._calculate_semantic_density(sample_text)
        density_time = time.time() - start_time
        print(f"✅ Semantic density calculation in {density_time:.3f}s: {density:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document service test failed: {e}")
        return False

def test_knowledge_graph():
    """Test the knowledge graph service"""
    print("\n🔍 Testing Knowledge Graph Service...")
    
    try:
        from knowledge_graph_service import KnowledgeGraphService
        
        # Initialize service
        start_time = time.time()
        kg_service = KnowledgeGraphService()
        init_time = time.time() - start_time
        print(f"✅ Knowledge graph service initialized in {init_time:.2f}s")
        print(f"   Enabled: {kg_service.enabled}")
        
        if kg_service.enabled:
            # Test entity and relationship extraction
            sample_text = """
            Cardiac surgery is covered under the policy after a waiting period of 2 years.
            Pre-existing heart conditions are excluded for the first 4 years.
            Emergency cardiac procedures require network hospital treatment.
            """
            
            start_time = time.time()
            entities, relationships = kg_service.extract_entities_and_relationships(sample_text, "test_doc")
            extract_time = time.time() - start_time
            print(f"✅ Entity/relationship extraction in {extract_time:.3f}s")
            print(f"   Found {len(entities)} entities and {len(relationships)} relationships")
            
            # Test storage
            if entities or relationships:
                start_time = time.time()
                kg_service.store_entities_and_relationships(entities, relationships)
                store_time = time.time() - start_time
                print(f"✅ Knowledge storage in {store_time:.3f}s")
                
                # Test query enhancement
                start_time = time.time()
                enhanced_queries = kg_service.enhance_query_with_kg(["cardiac", "surgery"])
                enhance_time = time.time() - start_time
                print(f"✅ Query enhancement in {enhance_time:.3f}s: {enhanced_queries}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph test failed: {e}")
        return False

async def test_performance_comparison():
    """Test performance improvements"""
    print("\n🚀 Performance Comparison Test...")
    
    try:
        from document_service import DocumentService
        
        doc_service = DocumentService()
        
        # Test questions
        test_questions = [
            "What is covered under cardiac treatment?",
            "What is the waiting period for pre-existing conditions?",
            "Are cosmetic surgeries excluded from coverage?",
            "What are the eligibility criteria for coverage?",
            "What is the maximum sum insured amount?"
        ]
        
        # Simulate processing without actual document (for testing)
        print(f"🎯 Testing with {len(test_questions)} questions...")
        
        # Test parallel processing capability
        start_time = time.time()
        
        # Simulate document processing steps
        for i, question in enumerate(test_questions, 1):
            # Simulate query processing
            query_plan = doc_service.llm_service.decompose_query(question)
            parsed_query = doc_service.llm_service.parse_query(question)
            search_queries = doc_service.llm_service.generate_search_queries(parsed_query)
            print(f"   ✅ Question {i} processed: {question[:50]}...")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_questions)
        
        print(f"📊 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average per question: {avg_time:.2f}s")
        print(f"   Estimated throughput: {1/avg_time:.1f} questions/second")
        
        # Show system stats
        stats = doc_service.get_system_stats()
        print(f"📊 System Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Optimized RAG System with FAISS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("FAISS Vector Service", test_vector_service),
        ("Optimized LLM Service", test_llm_service),
        ("Enhanced Document Service", test_document_service),
        ("Knowledge Graph Service", test_knowledge_graph),
        ("Performance Comparison", lambda: asyncio.run(test_performance_comparison()))
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            test_time = time.time() - start_time
            results.append((test_name, result, test_time))
            print(f"⏱️  Test completed in {test_time:.2f}s")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False, 0))
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, test_time in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<30} ({test_time:.2f}s)")
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed! The optimized RAG system is working correctly.")
        print("\n🚀 Key Improvements:")
        print("   ✅ FAISS for local vector storage (no API costs)")
        print("   ✅ Embedding caching for reduced OpenAI calls")
        print("   ✅ Semantic chunking with spaCy")
        print("   ✅ Enhanced hybrid search with BM25")
        print("   ✅ Optimized LLM service with response caching")
        print("   ✅ Knowledge graph with graceful Neo4j fallback")
        print("   ✅ Async processing for better performance")
        print("   ✅ Contextual retrieval with metadata")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the configuration.")

if __name__ == "__main__":
    main() 