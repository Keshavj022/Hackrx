#!/bin/bash
curl -X POST
"https://356370eea17f.ngrok-free.app/hackrx/run" \
-H "Authorization: Bearer 679b076ea66e474132c8ea9edcfd
3fd06a608834c6ab98900d1bec673ed9fe3c" \
-H "Content-Type: application/json" \
-d '{
    "documents": "https://hackrx.blob.core.windows.net/a
ssets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U1020
0WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%
3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzr
z1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing 
diseases?",
    "Does this policy cover maternity expenses?",
    "What is the sum insured amount?",
    "What are the exclusions in this policy?"
    ]
}'