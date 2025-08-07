# Advanced LLM-Powered Intelligent Query-Retrieval System

A state-of-the-art document processing system using advanced RAG (Retrieval-Augmented Generation) techniques to achieve >80% accuracy. Built for insurance, legal, HR, and compliance domains with hybrid search, query decomposition, reranking, and optional knowledge graph integration.

## Tech Stack

- **Backend**: FastAPI
- **Vector Database**: Pinecone  
- **LLM**: gpt-4o-mini
- **Database**: PostgreSQL
- **Document Processing**: PyPDF2, python-docx
- **Embeddings**: OpenAI text-embedding-3-small
- **Advanced RAG**: Sentence-Transformers, Cross-Encoders, BM25
- **Knowledge Graph**: Neo4j (Optional)

## Advanced Features

### 🔥 Hybrid Retrieval System
- **Dense Vector Search**: Semantic similarity using OpenAI embeddings
- **Sparse Keyword Search**: BM25-based retrieval for exact term matching
- **Combined Scoring**: Intelligent fusion of dense and sparse results

### 🧠 Query Decomposition & Multi-Step Reasoning
- **Intelligent Query Analysis**: Breaks complex questions into sub-questions
- **Entity Extraction**: Identifies key entities (procedures, amounts, conditions)
- **Reasoning Chains**: Step-by-step logical reasoning for complex queries

### 📊 Advanced Reranking
- **Cross-Encoder Reranking**: Uses cross-encoder/ms-marco-MiniLM-L-6-v2
- **Context-Aware Scoring**: Considers query-document relevance
- **Performance Optimization**: Balances accuracy and speed

### 🏗️ Hierarchical Document Chunking
- **Parent-Child Relationships**: Maintains document structure and context
- **Smart Section Detection**: Automatically identifies document sections
- **Context Preservation**: Links child chunks to parent context

### 🕸️ Knowledge Graph Integration (Optional)
- **Entity Relationship Modeling**: Maps relationships between concepts
- **Enhanced Query Expansion**: Uses graph traversal for better retrieval
- **Domain Knowledge**: Captures insurance-specific relationships

### 📈 Performance Optimizations
- **Token Efficiency**: Optimized prompt engineering for cost reduction
- **Response Caching**: Intelligent caching of processed documents
- **Parallel Processing**: Concurrent handling of multiple queries
- **Real-time Monitoring**: Built-in performance tracking and analytics

## Prerequisites

1. Python 3.8+
2. PostgreSQL (12+ recommended)
3. API Keys:
   - OpenAI API key with gpt-4o-mini access
   - Pinecone API key
4. Minimum 4GB RAM (8GB+ recommended)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_pinecone_index_name
   
   # PostgreSQL Configuration
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=your_database_name
   POSTGRES_USER=your_postgres_user
   POSTGRES_PASSWORD=your_postgres_password
   
   # API Configuration
   BEARER_TOKEN=your_secure_bearer_token
   
   # Advanced RAG Settings
   PARENT_CHUNK_SIZE=2048
   CHILD_CHUNK_SIZE=512
   USE_HYBRID_SEARCH=true
   USE_RERANKING=true
   USE_QUERY_DECOMPOSITION=true
   RERANK_TOP_K=20
   FINAL_TOP_K=8
   RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   
   # Neo4j Configuration (Optional - for Knowledge Graph)
   # NEO4J_URI=bolt://localhost:7687
   # NEO4J_USER=neo4j
   # NEO4J_PASSWORD=your_neo4j_password
   
   # Testing Configuration (optional)
   NGROK_URL=https://your-ngrok-url.ngrok-free.app
   API_BASE_URL=http://localhost:8000
   ```

3. Set up PostgreSQL database:
   ```bash
   # If using existing PostgreSQL installation (recommended)
   # Ensure PostgreSQL is running and you have a database created
   # The application will automatically create tables on first run
   
   # Or run the setup script (optional)
   python db_setup.py
   ```
   
   **Note**: If you're using Homebrew PostgreSQL on macOS, the default superuser is usually your system username, not `postgres`. Make sure your `.env` file reflects your actual PostgreSQL configuration.

## Quick Start

1. Start the server:
   ```bash
   python run_server.py
   ```
   
   Or directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Test the system:
   ```bash
   # Set up environment variables for testing
   export BEARER_TOKEN="your_bearer_token_here"
   export API_BASE_URL="http://localhost:8000"
   
   # Run the basic test suite
   python test_api.py
   
   # Run the advanced RAG test suite
   python test_advanced_rag.py
   
   # Or test manually with the shell script
   ./test.sh
   ```

3. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - System Stats: http://localhost:8000/stats

## API Usage

### Main Endpoint

**POST** `/hackrx/run`

Process documents and answer questions:

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer ${BEARER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

**Using ngrok for external testing:**
```bash
# Start ngrok (in a separate terminal)
ngrok http 8000

# Update your .env file with the ngrok URL
echo "NGROK_URL=https://your-ngrok-url.ngrok-free.app" >> .env

# Test with the provided script
./test.sh
```

### Response Format

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date...",
    "There is a waiting period of thirty-six (36) months of continuous coverage..."
  ]
}
```

## Project Structure

```
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── database.py          # PostgreSQL models and operations
├── vector_service.py    # Pinecone vector database service
├── llm_service.py       # gpt-4o-mini language model service
├── document_service.py  # Document processing orchestration
├── db_setup.py          # Database initialization
├── run_server.py        # Server startup script
├── test_api.py          # API test suite
├── requirements.txt     # Dependencies
├── .env.example         # Environment configuration template
├── data/                # Sample documents
│   ├── BAJHLIP23020V012223.pdf
│   ├── CHOTGDP23004V012223.pdf
│   ├── EDLHLGA23009V012223.pdf
│   ├── HDFHLIP23024V072223.pdf
│   └── ICIHLIP22012V012223.pdf
└── README.md           # Documentation
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │     gpt-4o-mini       │    │   Pinecone      │
│   Backend       │───▶│   LLM Service   │───▶│  Vector DB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        │
         │              ┌─────────────────┐                │
         │              │  Document       │                │
         │              │  Processing     │                │
         │              └─────────────────┘                │
         │                        │                        │
         ▼                        ▼                        │
┌─────────────────┐    ┌─────────────────┐                │
│   PostgreSQL    │    │     JSON        │◀───────────────┘
│   Database      │    │   Response      │
└─────────────────┘    └─────────────────┘
```

## Processing Workflow

1. Document URL received via API
2. Text extracted and chunked
3. Embeddings generated via OpenAI
4. Vectors stored in Pinecone
5. Questions parsed by gpt-4o-mini
6. Relevant chunks retrieved via vector search
7. gpt-4o-mini generates contextual answers
8. Structured JSON response returned

## Performance Features

- Intelligent caching for repeated queries
- Efficient batch processing of multiple questions
- Real-time performance monitoring and analytics
- Comprehensive health checks
- Robust error handling with fallback mechanisms

## Sample Usage

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?",
    "What is the No Claim Discount offered?",
    "Are there sub-limits on room rent for Plan A?"
  ]
}
```

## Domain Applications

- **Insurance**: Policy analysis, claim processing, coverage determination
- **Legal**: Contract review, compliance checking, clause extraction
- **HR**: Employee handbook queries, policy interpretation
- **Healthcare**: Benefits analysis, coverage verification

## Performance Targets

- **Accuracy**: 70%+ on complex policy questions
- **Latency**: <2 minutes for document processing + 5 questions
- **Scalability**: Handle multiple concurrent requests
- **Reliability**: 99.9% uptime with proper infrastructure

## Development & Testing

```bash
# Set environment variables first
source .env

# Run tests
python test_api.py

# Check system health
curl http://localhost:8000/health

# Get performance metrics
curl -H "Authorization: Bearer ${BEARER_TOKEN}" http://localhost:8000/stats

# Test with shell script (requires NGROK_URL in .env)
./test.sh
```

## Troubleshooting

**Common Issues:**

1. **Database Connection**: Ensure PostgreSQL is running and credentials are correct
   - On macOS with Homebrew: Check that your PostgreSQL user matches your system username
   - Verify database exists: `psql -l` to list databases
   
2. **Environment Variables**: All required variables must be set in `.env`
   - Copy from `.env.example` and fill in your actual values
   - Required: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`, `POSTGRES_*`, `BEARER_TOKEN`
   
3. **Pinecone Errors**: Verify API key and index configuration
   - Check Pinecone dashboard for correct environment and index name
   - Ensure index dimension matches embedding model (1536 for text-embedding-3-small)
   
4. **gpt-4o-mini Limits**: Check OpenAI account limits and billing
   
5. **Memory Issues**: Monitor RAM usage during document processing
   
6. **Bearer Token Authentication**: Make sure `BEARER_TOKEN` is set correctly in both `.env` and when making requests

**Performance Optimization:**

- Use SSD storage for PostgreSQL
- Configure Pinecone index properly
- Monitor OpenAI API usage
- Implement request rate limiting

## Monitoring

Built-in monitoring includes:

- Request/response times
- Success/failure rates  
- Vector database statistics
- gpt-4o-mini usage metrics
- Database performance

## Security

- **Environment Variables**: All sensitive data (API keys, passwords, tokens) stored in `.env` file
- **Bearer Token Authentication**: Secure API access with configurable tokens
- **Input Validation**: Pydantic models for request/response validation
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries
- **Rate Limiting**: Configurable request limits
- **Audit Logging**: Performance and usage tracking in PostgreSQL

**Security Best Practices:**
- Never commit `.env` files to version control
- Use strong, unique bearer tokens
- Regularly rotate API keys
- Monitor usage and access logs
- Use HTTPS in production environments

## License

This project is developed for the HackRx competition. Ensure compliance with OpenAI's usage policies and Pinecone's terms of service.