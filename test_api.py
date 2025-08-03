#!/usr/bin/env python3
"""Test suite for the document query-retrieval API"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "679b076ea66e474132c8ea9edcfd3fd06a608834c6ab98900d1bec673ed9fe3c"

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

def test_health_check():
    """Test API health status"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        result = response.json()
        
        if response.status_code == 200 and result['status'] in ['healthy', 'degraded']:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_authentication():
    """Test bearer token authentication"""
    invalid_headers = {"Authorization": "Bearer invalid_token", "Content-Type": "application/json"}
    
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=invalid_headers, timeout=10)
        if response.status_code == 401:
            print("✓ Authentication working")
            return True
        else:
            print("✗ Authentication bypass detected")
            return False
    except Exception as e:
        print(f"✗ Authentication test error: {e}")
        return False

def test_document_processing():
    """Test document processing with sample questions"""
    sample_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=sample_request, timeout=180)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answers_count = len(result['answers'])
            
            if answers_count == len(sample_request['questions']):
                print(f"✓ Document processing completed ({processing_time:.1f}s, {answers_count} answers)")
                return True
            else:
                print(f"✗ Answer count mismatch: expected {len(sample_request['questions'])}, got {answers_count}")
                return False
        else:
            print(f"✗ Processing failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Processing timeout")
        return False
    except Exception as e:
        print(f"✗ Processing error: {e}")
        return False

def test_system_stats():
    """Test system statistics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/stats", headers=headers, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            health_status = result.get('health_check', {})
            
            if all(health_status.values()):
                print("✓ System stats retrieved, all components healthy")
                return True
            else:
                print("⚠ System stats retrieved, some components unhealthy")
                return True  # Still pass if we can get stats
        else:
            print(f"✗ Stats retrieval failed: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Stats test error: {e}")
        return False

def run_tests():
    """Execute all test cases"""
    print("Running API test suite...")
    print("-" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Authentication", test_authentication), 
        ("System Stats", test_system_stats),
        ("Document Processing", test_document_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("Test Results:")
    print("-" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)