#!/bin/bash

# Load environment variables
source .env

# Check if BEARER_TOKEN is set
if [ -z "$BEARER_TOKEN" ]; then
    echo "Error: BEARER_TOKEN environment variable is not set"
    echo "Please set it in your .env file or export it"
    exit 1
fi

# Check if NGROK_URL is set
if [ -z "$NGROK_URL" ]; then
    echo "Error: NGROK_URL environment variable is not set"
    echo "Please set it in your .env file or export it"
    echo "Example: export NGROK_URL=https://your-ngrok-url.ngrok-free.app"
    exit 1
fi

curl -X POST "${NGROK_URL}/hackrx/run" \
-H "Authorization: Bearer ${BEARER_TOKEN}" \
-H "Content-Type: application/json" \
-d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?",
        "What is the sum insured amount?",
        "What are the exclusions in this policy?"
    ]
}'