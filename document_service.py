"""Document processing service for text extraction and vector storage"""

import os
import hashlib
import tempfile
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import logging
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from vector_service import VectorService
from llm_service import LLMService
from database import DatabaseManager

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
        return text
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from Word document"""
        text = ""
        try:
            doc = Document(docx_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX {docx_path}: {e}")
            raise
        return text
    
    def extract_text_from_email(self, email_path: str) -> str:
        """Extract text from email file"""
        text = ""
        try:
            with open(email_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            logger.error(f"Error reading email {email_path}: {e}")
            raise
        return text
    
    async def download_document(self, url: str) -> str:
        """Download document from URL and save to temporary file"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Determine file type from URL or content
                suffix = ".pdf"  # Default to PDF
                if url.lower().endswith('.docx'):
                    suffix = ".docx"
                elif url.lower().endswith('.txt'):
                    suffix = ".txt"
                elif url.lower().endswith('.eml'):
                    suffix = ".eml"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(response.content)
                    return tmp_file.name
                    
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            raise
    
    async def process_document_from_url(self, url: str) -> Dict[str, Any]:
        """Process a document from URL and store in vector database"""
        temp_file = None
        try:
            # Download document
            temp_file = await self.download_document(url)
            
            # Extract text based on file type
            if temp_file.endswith('.pdf'):
                text = self.extract_text_from_pdf(temp_file)
                file_type = "PDF"
            elif temp_file.endswith('.docx'):
                text = self.extract_text_from_docx(temp_file)
                file_type = "DOCX"
            else:
                text = self.extract_text_from_email(temp_file)
                file_type = "Text/Email"
            
            if not text.strip():
                raise ValueError("No text could be extracted from document")
            
            # Get file info
            file_size = os.path.getsize(temp_file)
            filename = os.path.basename(url).split('?')[0]  # Remove query params
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check if document already processed
            with DatabaseManager() as db:
                existing_doc = db.get_document_by_hash(content_hash)
                if existing_doc:
                    logger.info(f"Document already processed: {filename}")
                    return {
                        'document_id': existing_doc.id,
                        'chunks_processed': existing_doc.total_chunks,
                        'cached': True
                    }
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Prepare documents for Pinecone
            documents = []
            for i, chunk in enumerate(text_chunks):
                documents.append({
                    'content': chunk,
                    'source': filename,
                    'chunk_id': i,
                    'metadata': {'file_path': temp_file}
                })
            
            # Store in vector database
            vector_ids = self.vector_service.upsert_documents(documents)
            
            # Store in database
            with DatabaseManager() as db:
                # Save document metadata
                doc_record = db.save_document(
                    url=url,
                    filename=filename,
                    content_hash=content_hash,
                    total_chunks=len(documents),
                    file_size=file_size,
                    file_type=file_type
                )
                
                # Save document chunks
                for doc, vector_id in zip(documents, vector_ids):
                    db.save_document_chunk(
                        document_id=doc_record.id,
                        chunk_index=doc['chunk_id'],
                        content=doc['content'],
                        pinecone_id=vector_id,
                        chunk_metadata=doc['metadata']
                    )
                
                # Store doc_record.id before session closes
                document_id = doc_record.id
            
            return {
                'document_id': document_id,
                'chunks_processed': len(documents),
                'cached': False
            }
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        return self.vector_service.search_similar(query=query, top_k=top_k)
    
    async def process_question(self, question: str, document_url: str) -> str:
        """Process a single question against a document"""
        try:
            # Process document if not already done
            doc_info = await self.process_document_from_url(document_url)
            
            # Parse the question
            parsed_query = self.llm_service.parse_query(question)
            
            # Generate search queries
            search_queries = self.llm_service.generate_search_queries(parsed_query)
            
            # Search for relevant documents
            all_results = []
            for search_query in search_queries:
                results = self.search_documents(search_query, top_k=3)
                all_results.extend(results)
            
            # Remove duplicates and get top results
            unique_results = []
            seen_content = set()
            for result in all_results:
                content_hash = hashlib.md5(result['content'].encode()).hexdigest()
                if content_hash not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_hash)
            
            # Sort by relevance and take top results
            unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            top_results = unique_results[:settings.max_retrieval_docs]
            
            # Generate answer
            return self.llm_service.generate_direct_answer(question, top_results)
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Unable to process question due to error: {str(e)}"
    
    async def process_multiple_questions(self, questions: List[str], document_url: str) -> List[str]:
        """Process multiple questions against a document"""
        answers = []
        
        # Process document once
        try:
            await self.process_document_from_url(document_url)
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return [f"Error processing document: {str(e)}" for _ in questions]
        
        # Process each question
        for i, question in enumerate(questions):
            try:
                # Parse the question
                parsed_query = self.llm_service.parse_query(question)
                
                # Generate search queries
                search_queries = self.llm_service.generate_search_queries(parsed_query)
                
                # Search for relevant documents
                all_results = []
                for search_query in search_queries:
                    results = self.search_documents(search_query, top_k=3)
                    all_results.extend(results)
                
                # Remove duplicates and get top results
                unique_results = []
                seen_content = set()
                for result in all_results:
                    content_hash = hashlib.md5(result['content'].encode()).hexdigest()
                    if content_hash not in seen_content:
                        unique_results.append(result)
                        seen_content.add(content_hash)
                
                # Sort by relevance and take top results
                unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
                top_results = unique_results[:settings.max_retrieval_docs]
                
                # Generate answer
                answer = self.llm_service.generate_direct_answer(question, top_results)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Unable to process question due to error: {str(e)}")
        
        return answers
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.vector_service.get_index_stats()
            
            with DatabaseManager() as db:
                performance_stats = db.get_performance_stats(24)
            
            return {
                'vector_stats': vector_stats,
                'performance_stats': performance_stats,
                'llm_healthy': self.llm_service.health_check(),
                'vector_healthy': self.vector_service.health_check()
            }
        except Exception:
            return {}
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        return {
            'llm': self.llm_service.health_check(),
            'vector': self.vector_service.health_check(),
            'database': True
        }