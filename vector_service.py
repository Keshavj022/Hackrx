"""Vector database service for document embeddings and similarity search"""

import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
import logging
import hashlib
import time
from config import settings

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        self.pc = None
        self.index = None
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self._initialize()
    
    def _initialize(self):
        """Initialize vector database connection and ensure index exists"""
        try:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if settings.pinecone_index_name not in existing_indexes:
                self.pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.pinecone_dimension,
                    metric=settings.pinecone_metric,
                    spec=ServerlessSpec(cloud='aws', region=settings.pinecone_environment)
                )
                time.sleep(10)  # Wait for index creation
            
            self.index = self.pc.Index(settings.pinecone_index_name)
            
        except Exception as e:
            logger.error(f"Vector database initialization failed: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Store document vectors in the database"""
        texts = [doc['content'] for doc in documents]
        embeddings = self.get_embeddings(texts)
        
        vectors = []
        vector_ids = []
        
        for doc, embedding in zip(documents, embeddings):
            content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            vector_id = f"{doc['source']}_{doc['chunk_id']}_{content_hash[:8]}"
            vector_ids.append(vector_id)
            
            metadata = {
                'source': doc['source'][:100],
                'chunk_id': doc['chunk_id'],
                'content': doc['content'][:1000],
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return vector_ids
    
    def search_similar(self, query: str, top_k: int = 5, 
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find documents similar to the query"""
        query_embedding = self.get_embeddings([query])[0]
        
        search_kwargs = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        if filter_metadata:
            search_kwargs['filter'] = filter_metadata
        
        results = self.index.query(**search_kwargs)
        
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                'id': match.id,
                'content': match.metadata.get('content', ''),
                'source': match.metadata.get('source', ''),
                'chunk_id': match.metadata.get('chunk_id', 0),
                'relevance_score': float(match.score),
                'metadata': match.metadata
            })
        
        return formatted_results
    
    def delete_by_source(self, source: str):
        """Remove all vectors from a specific document source"""
        self.index.delete(filter={'source': source})
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception:
            return {}
    
    def health_check(self) -> bool:
        """Check vector database connectivity"""
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False