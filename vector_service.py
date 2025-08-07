"""Enhanced Vector Service with Advanced RAG Techniques - Sentence-Window, Auto-Merging, Hybrid Search"""

import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
import os
import hashlib
import time
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    sentence_window: Optional[List[str]] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: Optional[List[str]] = None
    semantic_level: int = 0  # 0=sentence, 1=paragraph, 2=section

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int
    retrieval_method: str
    explanation: str

class AdvancedVectorService:
    def __init__(self):
        self.embedding_model = None
        self.cross_encoder = None
        self.faiss_index = None
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_metadata = {}
        self.bm25_index = None
        self.sentence_window_size = 3
        self.auto_merge_threshold = 0.8
        self.data_path = "data/"
        
        # Legacy compatibility for existing code
        self.index = None
        self.metadata_store = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.dimension = settings.pinecone_dimension if hasattr(settings, 'pinecone_dimension') else 384
        self.document_corpus = []
        self.embedding_cache = {}
        
        # Create data directory if not exists
        os.makedirs(self.data_path, exist_ok=True)
        
    async def initialize(self):
        try:
            # Initialize embedding models
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize cross-encoder for re-ranking
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder initialized for re-ranking")
            except Exception as e:
                logger.warning(f"Cross-encoder not available: {e}")
                
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            # Initialize legacy FAISS index for compatibility
            self.index = faiss.IndexFlatIP(self.dimension)
                
            # Load existing indices if available
            await self._load_indices()
            
            logger.info("Advanced Vector Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            raise

    async def process_document_with_advanced_chunking(
        self, 
        document_content: str, 
        document_url: str,
        use_sentence_window: bool = True,
        use_auto_merging: bool = True
    ):
        try:
            logger.info("Starting advanced document processing with hierarchical chunking")
            
            # Step 1: Semantic chunking with multiple levels
            chunks = await self._create_hierarchical_chunks(document_content, document_url)
            
            # Step 2: Apply sentence-window technique
            if use_sentence_window:
                chunks = await self._apply_sentence_windowing(chunks, document_content)
            
            # Step 3: Create embeddings for all chunks
            await self._create_embeddings_batch(chunks)
            
            # Step 4: Build FAISS index with error handling
            try:
                await self._build_faiss_index(chunks)
            except Exception as e:
                logger.error(f"FAISS index creation failed: {e}")
                # Continue without FAISS - will fall back to simpler search
                self.faiss_index = None
                self.index = None
            
            # Step 5: Build BM25 index for sparse retrieval
            try:
                await self._build_bm25_index(chunks)
            except Exception as e:
                logger.warning(f"BM25 index creation failed, continuing without sparse search: {e}")
            
            # Step 6: Apply auto-merging if enabled (skip for now to avoid complexity)
            if use_auto_merging and len(chunks) < 500:  # Only for smaller documents
                try:
                    await self._apply_auto_merging(chunks)
                except Exception as e:
                    logger.warning(f"Auto-merging failed, continuing without: {e}")
            
            # Step 7: Save indices
            try:
                await self._save_indices()
            except Exception as e:
                logger.warning(f"Index saving failed, continuing: {e}")
            
            self.document_chunks = chunks
            logger.info(f"Advanced processing complete. Created {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error in advanced document processing: {str(e)}")
            raise

    async def _create_hierarchical_chunks(
        self, 
        document_content: str, 
        document_url: str
    ) -> List[DocumentChunk]:
        chunks = []
        
        # Level 0: Sentence-level chunks
        sentences = sent_tokenize(document_content)
        sentence_chunks = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Skip very short sentences
                chunk = DocumentChunk(
                    text=sentence.strip(),
                    metadata={
                        "document_url": document_url,
                        "chunk_type": "sentence",
                        "sentence_index": i,
                        "char_start": document_content.find(sentence),
                        "char_end": document_content.find(sentence) + len(sentence)
                    },
                    semantic_level=0
                )
                sentence_chunks.append(chunk)
                chunks.append(chunk)
        
        # Level 1: Paragraph-level chunks (3-5 sentences)
        paragraph_size = 4
        for i in range(0, len(sentence_chunks), paragraph_size):
            paragraph_sentences = sentence_chunks[i:i + paragraph_size]
            paragraph_text = " ".join([chunk.text for chunk in paragraph_sentences])
            
            if len(paragraph_text.strip()) > 50:
                paragraph_chunk = DocumentChunk(
                    text=paragraph_text,
                    metadata={
                        "document_url": document_url,
                        "chunk_type": "paragraph",
                        "paragraph_index": i // paragraph_size,
                        "sentence_range": (i, min(i + paragraph_size, len(sentence_chunks))),
                        "char_start": paragraph_sentences[0].metadata["char_start"],
                        "char_end": paragraph_sentences[-1].metadata["char_end"]
                    },
                    semantic_level=1,
                    child_chunk_ids=[str(id(chunk)) for chunk in paragraph_sentences]
                )
                
                # Set parent references
                for sent_chunk in paragraph_sentences:
                    sent_chunk.parent_chunk_id = str(id(paragraph_chunk))
                
                chunks.append(paragraph_chunk)
        
        # Level 2: Section-level chunks (larger semantic units)
        section_size = 3  # 3 paragraphs per section
        paragraph_chunks = [c for c in chunks if c.semantic_level == 1]
        
        for i in range(0, len(paragraph_chunks), section_size):
            section_paragraphs = paragraph_chunks[i:i + section_size]
            section_text = " ".join([chunk.text for chunk in section_paragraphs])
            
            if len(section_text.strip()) > 200:
                section_chunk = DocumentChunk(
                    text=section_text,
                    metadata={
                        "document_url": document_url,
                        "chunk_type": "section",
                        "section_index": i // section_size,
                        "paragraph_range": (i, min(i + section_size, len(paragraph_chunks))),
                        "char_start": section_paragraphs[0].metadata["char_start"],
                        "char_end": section_paragraphs[-1].metadata["char_end"]
                    },
                    semantic_level=2,
                    child_chunk_ids=[str(id(chunk)) for chunk in section_paragraphs]
                )
                
                # Set parent references
                for para_chunk in section_paragraphs:
                    para_chunk.parent_chunk_id = str(id(section_chunk))
                
                chunks.append(section_chunk)
        
        logger.info(f"Created hierarchical chunks: "
                   f"{len([c for c in chunks if c.semantic_level == 0])} sentences, "
                   f"{len([c for c in chunks if c.semantic_level == 1])} paragraphs, "
                   f"{len([c for c in chunks if c.semantic_level == 2])} sections")
        
        return chunks

    async def _apply_sentence_windowing(
        self, 
        chunks: List[DocumentChunk], 
        document_content: str
    ) -> List[DocumentChunk]:
        sentences = sent_tokenize(document_content)
        
        for chunk in chunks:
            if chunk.semantic_level == 0:  # Only apply to sentence-level chunks
                sentence_idx = chunk.metadata.get("sentence_index", 0)
                
                # Create window around current sentence
                start_idx = max(0, sentence_idx - self.sentence_window_size)
                end_idx = min(len(sentences), sentence_idx + self.sentence_window_size + 1)
                
                window_sentences = sentences[start_idx:end_idx]
                chunk.sentence_window = window_sentences
                
                # Create expanded context for better retrieval
                expanded_text = " ".join(window_sentences)
                chunk.metadata["windowed_text"] = expanded_text
                chunk.metadata["window_range"] = (start_idx, end_idx)
        
        logger.info("Applied sentence-window retrieval to sentence-level chunks")
        return chunks

    async def _create_embeddings_batch(self, chunks: List[DocumentChunk]):
        try:
            texts = []
            for chunk in chunks:
                # Use windowed text for sentence-level chunks, original text for others
                if chunk.semantic_level == 0 and chunk.sentence_window:
                    text = " ".join(chunk.sentence_window)
                else:
                    text = chunk.text
                
                # Truncate very long texts to prevent memory issues
                if len(text) > 2000:
                    text = text[:2000] + "..."
                texts.append(text)
            
            logger.info(f"Creating embeddings for {len(texts)} chunks")
            
            # Process embeddings in smaller batches to prevent memory issues
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i].astype('float32')
            
            logger.info("Embeddings created successfully")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            # Create dummy embeddings as fallback
            dummy_dimension = 384
            for chunk in chunks:
                chunk.embedding = np.random.random(dummy_dimension).astype('float32')
            logger.warning("Created dummy embeddings as fallback")

    async def _build_faiss_index(self, chunks: List[DocumentChunk]):
        try:
            embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
            dimension = embeddings.shape[1]
            
            logger.info(f"Building FAISS index with {len(chunks)} chunks, dimension {dimension}")
            
            # Use simpler IndexFlatIP for stability - avoid HNSW which can cause segfaults
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Also create legacy index for compatibility
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings in smaller batches to prevent memory issues
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                self.faiss_index.add(batch)
                self.index.add(batch)
                
                if i % 500 == 0:
                    logger.info(f"Added {i + len(batch)} vectors to FAISS index")
            
            logger.info(f"FAISS index built successfully with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            # Fallback to simple index
            try:
                dimension = embeddings.shape[1] if 'embeddings' in locals() else 384
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexFlatIP(dimension)
                logger.warning("Created empty FAISS index as fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback index: {fallback_error}")
                raise

    async def _build_bm25_index(self, chunks: List[DocumentChunk]):
        corpus = []
        self.document_corpus = []  # Reset corpus
        
        for chunk in chunks:
            # Tokenize text for BM25
            if chunk.semantic_level == 0 and chunk.sentence_window:
                text = " ".join(chunk.sentence_window)
            else:
                text = chunk.text
            
            tokens = word_tokenize(text.lower())
            corpus.append(tokens)
            self.document_corpus.append(text)
        
        self.bm25_index = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")

    async def _apply_auto_merging(self, chunks: List[DocumentChunk]):
        # Group chunks by semantic level
        chunks_by_level = defaultdict(list)
        for chunk in chunks:
            chunks_by_level[chunk.semantic_level].append(chunk)
        
        # Apply auto-merging at each level
        for level in chunks_by_level:
            level_chunks = chunks_by_level[level]
            
            # Calculate similarity matrix
            if len(level_chunks) > 1:
                embeddings = np.array([chunk.embedding for chunk in level_chunks])
                similarity_matrix = cosine_similarity(embeddings)
                
                # Find chunks to merge based on similarity threshold
                merged_groups = []
                visited = set()
                
                for i, chunk in enumerate(level_chunks):
                    if i in visited:
                        continue
                        
                    similar_chunks = [chunk]
                    visited.add(i)
                    
                    for j in range(i + 1, len(level_chunks)):
                        if j not in visited and similarity_matrix[i][j] > self.auto_merge_threshold:
                            similar_chunks.append(level_chunks[j])
                            visited.add(j)
                    
                    if len(similar_chunks) > 1:
                        merged_groups.append(similar_chunks)
                
                logger.info(f"Auto-merging: Found {len(merged_groups)} groups to merge at level {level}")

    async def advanced_similarity_search(
        self, 
        query: str, 
        k: int = 10,
        semantic_level: Optional[int] = None
    ) -> List[RetrievalResult]:
        try:
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                return []
                
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].astype('float32')
            query_vector = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(
                query_vector, 
                min(k * 2, len(self.document_chunks))  # Retrieve more for filtering
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    
                    # Filter by semantic level if specified
                    if semantic_level is not None and chunk.semantic_level != semantic_level:
                        continue
                    
                    result = RetrievalResult(
                        chunk=chunk,
                        score=float(score),
                        rank=i + 1,
                        retrieval_method="faiss_semantic",
                        explanation=f"Semantic similarity score: {score:.4f}"
                    )
                    results.append(result)
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def bm25_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        try:
            if not self.bm25_index:
                return []
            
            # Tokenize query
            query_tokens = word_tokenize(query.lower())
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            
            results = []
            for i, idx in enumerate(top_indices):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    score = bm25_scores[idx]
                    
                    result = RetrievalResult(
                        chunk=chunk,
                        score=float(score),
                        rank=i + 1,
                        retrieval_method="bm25_sparse",
                        explanation=f"BM25 relevance score: {score:.4f}"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []

    async def hybrid_search_with_rrf(
        self,
        query: str,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int = 10,
        rrf_constant: int = 60
    ) -> List[RetrievalResult]:
        try:
            # Reciprocal Rank Fusion (RRF)
            chunk_scores = defaultdict(float)
            chunk_methods = defaultdict(list)
            chunk_objects = {}
            
            # Process dense results
            for result in dense_results:
                chunk_id = str(id(result.chunk))
                chunk_scores[chunk_id] += 1.0 / (result.rank + rrf_constant)
                chunk_methods[chunk_id].append("dense")
                chunk_objects[chunk_id] = result.chunk
            
            # Process sparse results  
            for result in sparse_results:
                chunk_id = str(id(result.chunk))
                chunk_scores[chunk_id] += 1.0 / (result.rank + rrf_constant)
                chunk_methods[chunk_id].append("sparse")
                chunk_objects[chunk_id] = result.chunk
            
            # Sort by fused score
            sorted_chunks = sorted(
                chunk_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Create hybrid results
            hybrid_results = []
            for i, (chunk_id, score) in enumerate(sorted_chunks[:k]):
                chunk = chunk_objects[chunk_id]
                methods = "+".join(chunk_methods[chunk_id])
                
                result = RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=i + 1,
                    retrieval_method=f"hybrid_rrf_{methods}",
                    explanation=f"RRF hybrid score: {score:.4f} (methods: {methods})"
                )
                hybrid_results.append(result)
            
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search with RRF: {str(e)}")
            return dense_results[:k]  # Fallback to dense results

    async def apply_sentence_window_retrieval(
        self, 
        retrieval_results: List[RetrievalResult],
        window_size: int = 3
    ) -> List[RetrievalResult]:
        try:
            enhanced_results = []
            
            for result in retrieval_results:
                chunk = result.chunk
                
                if chunk.sentence_window and len(chunk.sentence_window) > 1:
                    # Use pre-computed sentence window
                    enhanced_text = " ".join(chunk.sentence_window)
                    
                    # Create enhanced chunk
                    enhanced_chunk = DocumentChunk(
                        text=enhanced_text,
                        metadata={
                            **chunk.metadata,
                            "enhanced_with": "sentence_window",
                            "original_text": chunk.text,
                            "window_sentences": len(chunk.sentence_window)
                        },
                        embedding=chunk.embedding,
                        sentence_window=chunk.sentence_window,
                        semantic_level=chunk.semantic_level
                    )
                    
                    enhanced_result = RetrievalResult(
                        chunk=enhanced_chunk,
                        score=result.score * 1.1,  # Slight boost for windowed content
                        rank=result.rank,
                        retrieval_method=result.retrieval_method + "+window",
                        explanation=result.explanation + f" | Enhanced with {len(chunk.sentence_window)} sentence window"
                    )
                    enhanced_results.append(enhanced_result)
                else:
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error applying sentence window retrieval: {str(e)}")
            return retrieval_results

    async def apply_auto_merging_retrieval(
        self, 
        retrieval_results: List[RetrievalResult],
        similarity_threshold: float = 0.8
    ) -> List[RetrievalResult]:
        try:
            if len(retrieval_results) <= 1:
                return retrieval_results
            
            # Group similar results
            merged_groups = []
            visited = set()
            
            for i, result1 in enumerate(retrieval_results):
                if i in visited:
                    continue
                    
                similar_group = [result1]
                visited.add(i)
                
                for j in range(i + 1, len(retrieval_results)):
                    if j in visited:
                        continue
                        
                    result2 = retrieval_results[j]
                    
                    # Calculate similarity between chunks
                    if result1.chunk.embedding is not None and result2.chunk.embedding is not None:
                        similarity = cosine_similarity(
                            [result1.chunk.embedding], 
                            [result2.chunk.embedding]
                        )[0][0]
                        
                        if similarity > similarity_threshold:
                            similar_group.append(result2)
                            visited.add(j)
                
                merged_groups.append(similar_group)
            
            # Create merged results
            final_results = []
            for group in merged_groups:
                if len(group) == 1:
                    final_results.append(group[0])
                else:
                    # Merge similar chunks
                    merged_text = " | ".join([r.chunk.text for r in group])
                    best_score = max([r.score for r in group])
                    best_rank = min([r.rank for r in group])
                    methods = "+".join(set([r.retrieval_method for r in group]))
                    
                    merged_chunk = DocumentChunk(
                        text=merged_text,
                        metadata={
                            "merged_from": len(group),
                            "original_chunks": [r.chunk.metadata for r in group],
                            "merge_method": "auto_merging"
                        },
                        semantic_level=group[0].chunk.semantic_level
                    )
                    
                    merged_result = RetrievalResult(
                        chunk=merged_chunk,
                        score=best_score * 1.15,  # Boost for merged relevance
                        rank=best_rank,
                        retrieval_method=methods + "+merged",
                        explanation=f"Auto-merged {len(group)} similar chunks (similarity > {similarity_threshold})"
                    )
                    final_results.append(merged_result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in auto-merging retrieval: {str(e)}")
            return retrieval_results

    async def rerank_contexts(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        try:
            if not self.cross_encoder or len(retrieval_results) <= 1:
                return retrieval_results
            
            # Prepare query-passage pairs
            pairs = []
            for result in retrieval_results:
                pairs.append([query, result.chunk.text[:512]])  # Limit text length
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Create reranked results
            reranked_results = []
            for i, (result, cross_score) in enumerate(zip(retrieval_results, cross_scores)):
                reranked_result = RetrievalResult(
                    chunk=result.chunk,
                    score=float(cross_score),
                    rank=i + 1,
                    retrieval_method=result.retrieval_method + "+reranked",
                    explanation=result.explanation + f" | Cross-encoder score: {cross_score:.4f}"
                )
                reranked_results.append(reranked_result)
            
            # Sort by cross-encoder scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            logger.info(f"Reranked {len(reranked_results)} results using cross-encoder")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return retrieval_results

    async def hybrid_search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        # Get dense and sparse results
        dense_task = self.advanced_similarity_search(query, k)
        sparse_task = self.bm25_search(query, k)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Combine with RRF
        hybrid_results = await self.hybrid_search_with_rrf(
            query, dense_results, sparse_results, k
        )
        
        return hybrid_results

    async def _save_indices(self):
        try:
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, f"{self.data_path}faiss_index.bin")
            
            # Save legacy FAISS index for compatibility
            if self.index:
                faiss.write_index(self.index, f"{self.data_path}faiss_index_legacy.bin")
            
            # Save document chunks
            with open(f"{self.data_path}document_corpus.pkl", 'wb') as f:
                pickle.dump(self.document_chunks, f)
            
            # Save BM25 index
            if self.bm25_index:
                with open(f"{self.data_path}bm25_index.pkl", 'wb') as f:
                    pickle.dump(self.bm25_index, f)
            
            logger.info("Advanced indices saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving indices: {str(e)}")

    async def _load_indices(self):
        try:
            # Load FAISS index
            faiss_path = f"{self.data_path}faiss_index.bin"
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                logger.info("Advanced FAISS index loaded")
            
            # Load legacy FAISS index for compatibility
            legacy_faiss_path = f"{self.data_path}faiss_index_legacy.bin"
            if os.path.exists(legacy_faiss_path):
                self.index = faiss.read_index(legacy_faiss_path)
                logger.info("Legacy FAISS index loaded")
            
            # Load document chunks
            corpus_path = f"{self.data_path}document_corpus.pkl"
            if os.path.exists(corpus_path):
                with open(corpus_path, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logger.info(f"Document corpus loaded: {len(self.document_chunks)} chunks")
            
            # Load BM25 index
            bm25_path = f"{self.data_path}bm25_index.pkl"
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info("BM25 index loaded")
                
        except Exception as e:
            logger.warning(f"Could not load existing indices: {str(e)}")

    def is_healthy(self) -> bool:
        return (
            self.embedding_model is not None and
            (self.faiss_index is not None or self.index is not None)
        )

    # Legacy compatibility methods
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        # Compatibility method for existing code
        results = await self.advanced_similarity_search(query, k)
        return [
            {
                "text": result.chunk.text,
                "metadata": result.chunk.metadata,
                "score": result.score,
                "method": result.retrieval_method
            }
            for result in results
        ]

    def health_check(self) -> bool:
        return self.is_healthy()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer for compatibility"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    def upsert_documents(self, documents: List[Dict[str, Any]], 
                        parent_chunks: Optional[Dict[str, str]] = None) -> List[str]:
        """Legacy method for backward compatibility"""
        logger.warning("Using legacy upsert_documents method. Consider using process_document_with_advanced_chunking for better performance")
        
        vector_ids = []
        for doc in documents:
            content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            vector_id = f"{doc.get('source', 'unknown')}_{doc.get('chunk_id', 0)}_{content_hash[:8]}"
            vector_ids.append(vector_id)
            
            # Create a simple chunk for compatibility
            chunk = DocumentChunk(
                text=doc['content'],
                metadata={
                    'source': doc.get('source', 'unknown'),
                    'chunk_id': doc.get('chunk_id', 0),
                    'content': doc['content']
                },
                semantic_level=0
            )
            self.document_chunks.append(chunk)
        
        return vector_ids

    def search_similar(self, query: str, top_k: int = 5, 
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Legacy search method for backward compatibility"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self.similarity_search(query, top_k))
            loop.close()
            return results
        except Exception as e:
            logger.error(f"Error in legacy search: {str(e)}")
            return []

# Alias for backward compatibility
VectorService = AdvancedVectorService