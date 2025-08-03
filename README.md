# Document Query-Retrieval System

A document processing system that uses GPT-4, Pinecone vector database, and PostgreSQL to answer questions based on document content. Built for insurance, legal, HR, and compliance domains.

## Tech Stack

- **Backend**: FastAPI
- **Vector Database**: Pinecone  
- **LLM**: GPT-4
- **Database**: PostgreSQL
- **Document Processing**: PyPDF2, python-docx
- **Embeddings**: OpenAI text-embedding-3-small

## Features

- GPT-4 powered document analysis and question answering
- Pinecone vector search for semantic document retrieval
- PostgreSQL for data persistence and performance tracking
- Multi-format support (PDF, DOCX, text files)
- Real-time processing via document URLs
- Built-in performance monitoring and health checks
- Bearer token authentication

## Prerequisites

1. Python 3.8+
2. PostgreSQL (12+ recommended)
3. API Keys:
   - OpenAI API key with GPT-4 access
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
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   POSTGRES_PASSWORD=your_postgres_password_here
   ```

3. Set up PostgreSQL database:
   ```bash
   python db_setup.py
   ```

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
   python test_api.py
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
  -H "Authorization: Bearer 679b076ea66e474132c8ea9edcfd3fd06a608834c6ab98900d1bec673ed9fe3c" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
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
├── llm_service.py       # GPT-4 language model service
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
│   FastAPI       │    │     GPT-4       │    │   Pinecone      │
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
5. Questions parsed by GPT-4
6. Relevant chunks retrieved via vector search
7. GPT-4 generates contextual answers
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
# Run tests
python test_api.py

# Check system health
curl http://localhost:8000/health

# Get performance metrics
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/stats
```

## Troubleshooting

**Common Issues:**

1. **Database Connection**: Ensure PostgreSQL is running and credentials are correct
2. **Pinecone Errors**: Verify API key and index configuration
3. **GPT-4 Limits**: Check OpenAI account limits and billing
4. **Memory Issues**: Monitor RAM usage during document processing

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
- GPT-4 usage metrics
- Database performance

## Security

- Bearer token authentication
- Input validation and sanitization
- SQL injection protection
- Rate limiting capabilities
- Audit logging

## License

This project is developed for the HackRx competition. Ensure compliance with OpenAI's usage policies and Pinecone's terms of service.