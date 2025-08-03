"""Language model service for query processing and answer generation"""

import openai
from typing import Dict, List, Any
import json
import logging
import re
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query"""
        prompt = f"""Parse this query and extract key information. Return only valid JSON.

Extract these fields if present:
- age: person's age (number)
- gender: M/F/Male/Female  
- procedure: medical procedure or treatment
- location: city or location
- policy_duration: how long policy has been active
- policy_type: type of insurance policy
- amount: any monetary amount mentioned
- condition: medical condition or diagnosis
- urgency: urgent/emergency/routine
- date: any relevant dates

Query: "{query}"

Return only a JSON object with the extracted information. If a field is not present, omit it from the JSON.

JSON:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at parsing queries. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
            
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return self._fallback_parse(query)
    
    def generate_search_queries(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Create multiple search queries to find relevant document sections"""
        prompt = f"""Based on this parsed query information, generate 3-5 specific search queries.

Parsed Query: {json.dumps(parsed_query, indent=2)}

Generate search queries that cover:
1. The specific procedure/condition coverage
2. Waiting periods and eligibility requirements
3. Exclusions and limitations
4. Network restrictions if location is mentioned
5. Policy-specific terms related to the query

Return only a JSON array of search query strings.

JSON:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Generate search queries for document retrieval. Return valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
            
        except Exception as e:
            logger.error(f"Search query generation failed: {e}")
            return self._fallback_search_queries(parsed_query)
    
    def evaluate_claim(self, parsed_query: Dict[str, Any], relevant_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate insurance claim using GPT-4"""
        context = self._prepare_context(relevant_documents)
        
        prompt = f"""You are an expert insurance claim evaluator. Analyze this claim and return your decision in JSON format.

Consider these factors:
- Coverage eligibility
- Waiting periods  
- Pre-existing conditions
- Policy duration
- Exclusions
- Network restrictions
- Sum insured limits

Query Details:
{json.dumps(parsed_query, indent=2)}

Relevant Policy Information:
{context}

Analyze the claim and return JSON in this exact format:
{{
    "decision": "approved" or "rejected",
    "amount": numeric_value_or_null,
    "confidence": 0.85,
    "justification": "detailed explanation of the decision",
    "relevant_clauses": ["clause1", "clause2"],
    "reasoning_steps": ["step1", "step2", "step3"]
}}

JSON:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert insurance claim evaluator. Always return valid JSON with accurate decisions based on policy terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = json.loads(content)
            
            # Validate and clean decision
            decision = self._validate_decision(decision)
            logger.info("Successfully evaluated claim with GPT-4")
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating claim with GPT-4: {e}")
            return self._fallback_decision(parsed_query, relevant_documents)
    
    def generate_direct_answer(self, question: str, relevant_documents: List[Dict[str, Any]]) -> str:
        """Generate answer to question based on document context"""
        context = self._prepare_context(relevant_documents)
        
        prompt = f"""Answer the following question based on the provided document context.

Question: {question}

Document Context:
{context}

Instructions:
- Provide a clear, direct answer
- Include specific details from the documents
- If information is not available, state so clearly
- Focus on factual information from the source material
- Be concise but comprehensive

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analyst. Provide accurate answers based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._fallback_answer(question, relevant_documents)
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format document content for model context"""
        context = ""
        for i, doc in enumerate(documents, 1):
            context += f"\n--- Document {i} (Source: {doc['source']}) ---\n"
            content = doc['content'][:2000]
            if len(doc['content']) > 2000:
                content += "...[truncated]"
            context += content + "\n"
        return context
    
    def _validate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the decision output"""
        # Ensure required fields exist
        decision.setdefault('decision', 'rejected')
        decision.setdefault('amount', None)
        decision.setdefault('confidence', 0.5)
        decision.setdefault('justification', 'Decision made based on available information')
        decision.setdefault('relevant_clauses', [])
        decision.setdefault('reasoning_steps', [])
        
        # Validate decision value
        if decision['decision'] not in ['approved', 'rejected']:
            decision['decision'] = 'rejected'
        
        # Validate confidence
        try:
            confidence = float(decision['confidence'])
            decision['confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            decision['confidence'] = 0.5
        
        # Validate amount
        if decision['amount'] is not None:
            try:
                decision['amount'] = float(decision['amount'])
            except (ValueError, TypeError):
                decision['amount'] = None
        
        return decision
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Basic regex-based query parsing when LLM fails"""
        parsed = {}
        
        age_match = re.search(r'(\d+)[-\s]*(year|yr|y)?[-\s]*(old|M|F)', query, re.IGNORECASE)
        if age_match:
            parsed['age'] = int(age_match.group(1))
        
        if re.search(r'\b(male|M)\b', query, re.IGNORECASE):
            parsed['gender'] = 'M'
        elif re.search(r'\b(female|F)\b', query, re.IGNORECASE):
            parsed['gender'] = 'F'
        
        procedures = ['surgery', 'operation', 'treatment', 'procedure']
        for proc in procedures:
            if proc.lower() in query.lower():
                parsed['procedure'] = proc
                break
        
        return parsed
    
    def _fallback_search_queries(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Generate basic search queries when LLM fails"""
        queries = []
        if 'procedure' in parsed_query:
            queries.extend([f"coverage {parsed_query['procedure']}", f"waiting period {parsed_query['procedure']}"])
        queries.extend(["exclusions", "policy terms", "coverage limits"])
        return queries
    
    def _fallback_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Return basic answer when LLM fails"""
        if not documents:
            return "No relevant information found in the documents to answer this question."
        
        best_doc = max(documents, key=lambda x: x.get('relevance_score', 0))
        return f"Based on the available information: {best_doc['content'][:300]}..."
    
    def health_check(self) -> bool:
        """Check LLM API connectivity"""
        try:
            self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception:
            return False