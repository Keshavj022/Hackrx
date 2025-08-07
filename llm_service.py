"""Enhanced LLM Service with Advanced RAG Triad Evaluation and Ollama Integration"""

import asyncio
import logging
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import hashlib
import pickle
import os
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class RAGTriadScores:
    context_relevance: float
    groundedness: float
    answer_relevance: float
    confidence: float
    explanation: str

class AdvancedLLMService:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.default_model = "llama3.1:8b"
        self.available_models = []
        self.token_count = 0
        self.total_requests = 0
        
        # Response caching for efficiency
        self.response_cache = {}
        self.cache_file = "data/llm_response_cache.pkl"
        self._load_cache()
        
        # Legacy OpenAI compatibility
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=getattr(settings, 'openai_api_key', None))
            self.use_openai_fallback = bool(getattr(settings, 'openai_api_key', None))
        except:
            self.openai_client = None
            self.use_openai_fallback = False
        
    def _load_cache(self):
        """Load response cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.response_cache = pickle.load(f)
                logger.info(f"Loaded LLM cache with {len(self.response_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load LLM cache: {e}")
    
    def _save_cache(self):
        """Save response cache to disk"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.response_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")
        
    async def initialize(self):
        try:
            # Check Ollama availability and get available models
            await self._check_ollama_health()
            await self._get_available_models()
            logger.info(f"Advanced LLM Service initialized with models: {self.available_models}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            if self.use_openai_fallback:
                logger.info("Falling back to OpenAI API")
            else:
                raise

    async def _check_ollama_health(self):
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama service is healthy")
                return True
            else:
                raise Exception(f"Ollama health check failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            if not self.use_openai_fallback:
                raise

    async def _get_available_models(self):
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                
                # Check if our preferred model is available
                if self.default_model not in self.available_models:
                    if self.available_models:
                        self.default_model = self.available_models[0]
                        logger.warning(f"Default model not found, using: {self.default_model}")
                    else:
                        raise Exception("No models available in Ollama")
                        
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            self.available_models = [self.default_model]  # Fallback

    async def _call_ollama(self, prompt: str, model: str = None, system_prompt: str = None) -> Dict[str, Any]:
        try:
            model = model or self.default_model
            self.total_requests += 1
            
            # Check cache first
            cache_key = self._get_cache_key(prompt, model, system_prompt)
            if cache_key in self.response_cache:
                logger.debug("Cache hit for LLM call")
                return self.response_cache[cache_key]
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent evaluation
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Track token usage if available
                if 'eval_count' in result:
                    self.token_count += result.get('eval_count', 0)
                
                response_data = {
                    'response': result.get('response', ''),
                    'model': model,
                    'done': result.get('done', True),
                    'total_duration': result.get('total_duration', 0),
                    'eval_count': result.get('eval_count', 0)
                }
                
                # Cache the response
                self.response_cache[cache_key] = response_data
                
                # Save cache periodically
                if len(self.response_cache) % 20 == 0:
                    self._save_cache()
                
                return response_data
            else:
                logger.error(f"Ollama API call failed: {response.status_code} - {response.text}")
                return {'response': '', 'error': f"API call failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return {'response': '', 'error': str(e)}

    def _get_cache_key(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Generate cache key for responses"""
        content = f"{prompt}|{model}|{system_prompt or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _call_openai_fallback(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Fallback to OpenAI API if available"""
        try:
            if not self.use_openai_fallback:
                return {'response': '', 'error': 'OpenAI fallback not available'}
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=getattr(settings, 'openai_model', 'gpt-4o-mini'),
                messages=messages,
                max_tokens=800,
                temperature=0.1
            )
            
            return {
                'response': response.choices[0].message.content,
                'model': 'openai-fallback',
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {str(e)}")
            return {'response': '', 'error': str(e)}

    async def generate_answer_with_advanced_reasoning(
        self, 
        question: str, 
        contexts: List[Dict], 
        model: str = None
    ) -> Dict[str, Any]:
        try:
            # Prepare context text
            context_texts = []
            for i, ctx in enumerate(contexts[:5]):  # Limit to top 5 contexts
                if hasattr(ctx, 'chunk'):
                    text = ctx.chunk.text
                    method = getattr(ctx, 'retrieval_method', 'unknown')
                    score = getattr(ctx, 'score', 0.0)
                    context_texts.append(f"[Context {i+1}] (Method: {method}, Score: {score:.3f})\n{text}")
                elif isinstance(ctx, dict):
                    text = ctx.get('text', str(ctx))
                    method = ctx.get('method', 'unknown') 
                    score = ctx.get('score', 0.0)
                    context_texts.append(f"[Context {i+1}] (Method: {method}, Score: {score:.3f})\n{text}")
                else:
                    context_texts.append(f"[Context {i+1}]\n{str(ctx)}")
            
            context_text = "\n\n".join(context_texts)
            
            # Create advanced reasoning prompt
            system_prompt = """You are an advanced document analysis AI that provides accurate, explainable answers based on retrieved context.

Your task is to:
1. Analyze the provided contexts carefully
2. Answer the question based only on the given information
3. Provide clear reasoning for your answer
4. Indicate confidence level and any limitations
5. Structure your response for maximum clarity

Always be precise, factual, and transparent about what information supports your answer."""

            reasoning_prompt = f"""Question: {question}

Retrieved Contexts:
{context_text}

Please provide a comprehensive answer following this structure:

ANSWER: [Your direct answer to the question]

REASONING: [Explain which contexts support your answer and how]

CONFIDENCE: [High/Medium/Low based on context quality and completeness]

LIMITATIONS: [Any important caveats or missing information]

Remember: Only use information from the provided contexts. If the contexts don't contain enough information to answer the question completely, clearly state this limitation."""

            # Try Ollama first, then fallback to OpenAI
            result = await self._call_ollama(reasoning_prompt, model, system_prompt)
            
            if 'error' in result or not result.get('response'):
                logger.warning("Ollama failed, trying OpenAI fallback")
                result = await self._call_openai_fallback(reasoning_prompt, system_prompt)
            
            if 'error' in result:
                logger.error(f"LLM generation failed: {result['error']}")
                return {
                    'answer': "I apologize, but I encountered an error while processing your question.",
                    'reasoning': result['error'],
                    'confidence': 0.0,
                    'model_used': model or self.default_model,
                    'contexts_used': len(contexts)
                }
            
            response_text = result.get('response', '')
            
            # Parse structured response
            parsed_response = self._parse_structured_response(response_text)
            
            return {
                'answer': parsed_response.get('answer', response_text),
                'reasoning': parsed_response.get('reasoning', ''),
                'confidence': parsed_response.get('confidence', 0.5),
                'limitations': parsed_response.get('limitations', ''),
                'model_used': result.get('model', model or self.default_model),
                'contexts_used': len(contexts),
                'generation_time': result.get('total_duration', 0) / 1000000000,  # Convert to seconds
                'tokens_generated': result.get('eval_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced answer generation: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question.",
                'reasoning': str(e),
                'confidence': 0.0,
                'model_used': model or self.default_model,
                'contexts_used': len(contexts)
            }

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        try:
            parsed = {}
            
            # Extract ANSWER
            answer_match = re.search(r'ANSWER:\s*(.+?)(?=\n\n|REASONING:|$)', response, re.DOTALL)
            if answer_match:
                parsed['answer'] = answer_match.group(1).strip()
            
            # Extract REASONING
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|CONFIDENCE:|$)', response, re.DOTALL)
            if reasoning_match:
                parsed['reasoning'] = reasoning_match.group(1).strip()
            
            # Extract CONFIDENCE
            confidence_match = re.search(r'CONFIDENCE:\s*(.+?)(?=\n\n|LIMITATIONS:|$)', response, re.DOTALL)
            if confidence_match:
                confidence_text = confidence_match.group(1).strip().lower()
                if 'high' in confidence_text:
                    parsed['confidence'] = 0.9
                elif 'medium' in confidence_text:
                    parsed['confidence'] = 0.7
                elif 'low' in confidence_text:
                    parsed['confidence'] = 0.4
                else:
                    parsed['confidence'] = 0.5
            
            # Extract LIMITATIONS
            limitations_match = re.search(r'LIMITATIONS:\s*(.+?)$', response, re.DOTALL)
            if limitations_match:
                parsed['limitations'] = limitations_match.group(1).strip()
            
            # If no structured format found, use the entire response as answer
            if not parsed.get('answer'):
                parsed['answer'] = response.strip()
                parsed['confidence'] = 0.5
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse structured response: {str(e)}")
            return {'answer': response.strip(), 'confidence': 0.5}

    async def evaluate_context_relevance(self, question: str, contexts: List[Dict]) -> float:
        try:
            # Prepare contexts for evaluation
            context_texts = []
            for i, ctx in enumerate(contexts[:5]):
                if hasattr(ctx, 'chunk'):
                    text = ctx.chunk.text[:300]  # Limit context length
                elif isinstance(ctx, dict):
                    text = ctx.get('text', str(ctx))[:300]
                else:
                    text = str(ctx)[:300]
                context_texts.append(f"Context {i+1}: {text}")
            
            context_text = "\n".join(context_texts)
            
            evaluation_prompt = f"""Question: {question}

Retrieved Contexts:
{context_text}

Evaluate how relevant these contexts are to answering the question. 

Rate the overall context relevance on a scale of 0.0 to 1.0 where:
- 1.0 = Highly relevant, contexts directly address the question
- 0.7-0.9 = Moderately relevant, contexts contain useful information
- 0.4-0.6 = Somewhat relevant, contexts have some related information
- 0.0-0.3 = Low relevance, contexts don't help answer the question

Provide your evaluation as a single number between 0.0 and 1.0, followed by a brief explanation.

Format: SCORE: [0.0-1.0]
EXPLANATION: [Brief reasoning]"""

            result = await self._call_ollama(evaluation_prompt)
            
            if 'error' in result or not result.get('response'):
                result = await self._call_openai_fallback(evaluation_prompt)
            
            response = result.get('response', '')
            
            # Extract score
            score_match = re.search(r'SCORE:\s*([0-1]\.?[0-9]*)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            else:
                # Fallback: analyze response for keywords
                response_lower = response.lower()
                if any(word in response_lower for word in ['highly relevant', 'very relevant', 'excellent']):
                    return 0.9
                elif any(word in response_lower for word in ['moderately relevant', 'good', 'useful']):
                    return 0.7
                elif any(word in response_lower for word in ['somewhat relevant', 'partially']):
                    return 0.5
                elif any(word in response_lower for word in ['low relevance', 'not relevant', 'poor']):
                    return 0.2
                else:
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Error evaluating context relevance: {str(e)}")
            return 0.5  # Default middle score

    async def evaluate_groundedness(self, contexts: List[Dict], answer: str) -> float:
        try:
            # Prepare contexts
            context_texts = []
            for i, ctx in enumerate(contexts[:5]):
                if hasattr(ctx, 'chunk'):
                    text = ctx.chunk.text[:300]
                elif isinstance(ctx, dict):
                    text = ctx.get('text', str(ctx))[:300]
                else:
                    text = str(ctx)[:300]
                context_texts.append(f"Context {i+1}: {text}")
            
            context_text = "\n".join(context_texts)
            
            evaluation_prompt = f"""Answer: {answer}

Source Contexts:
{context_text}

Evaluate how well the answer is grounded in (supported by) the provided contexts.

Rate the groundedness on a scale of 0.0 to 1.0 where:
- 1.0 = Answer is completely supported by the contexts
- 0.7-0.9 = Answer is mostly supported, minor unsupported elements
- 0.4-0.6 = Answer is partially supported by contexts
- 0.0-0.3 = Answer has little to no support from contexts

Check if the answer:
- Makes claims not found in the contexts
- Contradicts information in the contexts
- Stays within the bounds of provided information

Format: SCORE: [0.0-1.0]
EXPLANATION: [Brief reasoning about what is/isn't supported]"""

            result = await self._call_ollama(evaluation_prompt)
            
            if 'error' in result or not result.get('response'):
                result = await self._call_openai_fallback(evaluation_prompt)
                
            response = result.get('response', '')
            
            # Extract score
            score_match = re.search(r'SCORE:\s*([0-1]\.?[0-9]*)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            else:
                # Fallback keyword analysis
                response_lower = response.lower()
                if any(word in response_lower for word in ['completely supported', 'fully grounded']):
                    return 0.95
                elif any(word in response_lower for word in ['mostly supported', 'well grounded']):
                    return 0.8
                elif any(word in response_lower for word in ['partially supported', 'somewhat grounded']):
                    return 0.6
                elif any(word in response_lower for word in ['little support', 'not grounded', 'contradicts']):
                    return 0.3
                else:
                    return 0.6
                    
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {str(e)}")
            return 0.6

    async def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        try:
            evaluation_prompt = f"""Question: {question}

Answer: {answer}

Evaluate how well the answer addresses the specific question asked.

Rate the answer relevance on a scale of 0.0 to 1.0 where:
- 1.0 = Answer directly and completely addresses the question
- 0.7-0.9 = Answer addresses the question well with minor gaps
- 0.4-0.6 = Answer partially addresses the question
- 0.0-0.3 = Answer doesn't address the question or is off-topic

Consider:
- Does the answer directly respond to what was asked?
- Is the answer complete for the question type?
- Does the answer stay on topic?

Format: SCORE: [0.0-1.0]
EXPLANATION: [Brief reasoning about relevance]"""

            result = await self._call_ollama(evaluation_prompt)
            
            if 'error' in result or not result.get('response'):
                result = await self._call_openai_fallback(evaluation_prompt)
                
            response = result.get('response', '')
            
            # Extract score
            score_match = re.search(r'SCORE:\s*([0-1]\.?[0-9]*)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            else:
                # Fallback keyword analysis
                response_lower = response.lower()
                if any(word in response_lower for word in ['directly addresses', 'completely addresses']):
                    return 0.9
                elif any(word in response_lower for word in ['addresses well', 'good response']):
                    return 0.8
                elif any(word in response_lower for word in ['partially addresses', 'somewhat relevant']):
                    return 0.5
                elif any(word in response_lower for word in ['doesn\'t address', 'off-topic', 'irrelevant']):
                    return 0.2
                else:
                    return 0.6
                    
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return 0.6

    async def evaluate_rag_triad_comprehensive(
        self, 
        question: str, 
        contexts: List[Dict], 
        answer: str
    ) -> RAGTriadScores:
        try:
            # Run all three evaluations concurrently
            context_relevance_task = self.evaluate_context_relevance(question, contexts)
            groundedness_task = self.evaluate_groundedness(contexts, answer)
            answer_relevance_task = self.evaluate_answer_relevance(question, answer)
            
            context_relevance, groundedness, answer_relevance = await asyncio.gather(
                context_relevance_task, groundedness_task, answer_relevance_task
            )
            
            # Calculate overall confidence score (weighted average)
            confidence = (
                0.25 * context_relevance +   # How good are retrieved contexts
                0.40 * groundedness +        # How well grounded is the answer
                0.35 * answer_relevance      # How relevant is the answer
            )
            
            # Generate explanation
            explanation = self._generate_rag_triad_explanation(
                context_relevance, groundedness, answer_relevance, confidence
            )
            
            return RAGTriadScores(
                context_relevance=context_relevance,
                groundedness=groundedness,
                answer_relevance=answer_relevance,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive RAG triad evaluation: {str(e)}")
            return RAGTriadScores(
                context_relevance=0.5,
                groundedness=0.5, 
                answer_relevance=0.5,
                confidence=0.5,
                explanation=f"Evaluation failed: {str(e)}"
            )

    def _generate_rag_triad_explanation(
        self, 
        context_relevance: float, 
        groundedness: float, 
        answer_relevance: float,
        confidence: float
    ) -> str:
        explanations = []
        
        # Context Relevance
        if context_relevance >= 0.8:
            explanations.append("Retrieved contexts are highly relevant to the question")
        elif context_relevance >= 0.6:
            explanations.append("Retrieved contexts are moderately relevant")
        else:
            explanations.append("Retrieved contexts have limited relevance to the question")
        
        # Groundedness
        if groundedness >= 0.8:
            explanations.append("Answer is well-grounded in the provided contexts")
        elif groundedness >= 0.6:
            explanations.append("Answer is partially grounded in the contexts")
        else:
            explanations.append("Answer may contain information not supported by contexts")
        
        # Answer Relevance
        if answer_relevance >= 0.8:
            explanations.append("Answer directly addresses the question asked")
        elif answer_relevance >= 0.6:
            explanations.append("Answer partially addresses the question")
        else:
            explanations.append("Answer may not fully address the question")
        
        # Overall assessment
        if confidence >= 0.8:
            overall = "High confidence in response quality"
        elif confidence >= 0.6:
            overall = "Moderate confidence in response quality"
        else:
            overall = "Low confidence in response quality"
        
        return f"{overall}. {'. '.join(explanations)}."

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.token_count,
            'available_models': self.available_models,
            'default_model': self.default_model,
            'ollama_url': self.ollama_base_url,
            'cache_size': len(self.response_cache),
            'use_openai_fallback': self.use_openai_fallback
        }

    def is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            elif self.use_openai_fallback:
                # Test OpenAI fallback
                return self.openai_client is not None
            else:
                return False
        except:
            return self.use_openai_fallback

    # Legacy compatibility methods for backward compatibility
    async def generate_answer(self, question: str, contexts: List[Dict]) -> str:
        result = await self.generate_answer_with_advanced_reasoning(question, contexts)
        return result.get('answer', '')

    async def generate_direct_answer(self, question: str, relevant_documents: List[Dict[str, Any]]) -> str:
        """Legacy method for backward compatibility"""
        return await self.generate_answer(question, relevant_documents)

    def health_check(self) -> bool:
        """Legacy health check method"""
        return self.is_healthy()

# Alias for backward compatibility
LLMService = AdvancedLLMService