#!/usr/bin/env python3
"""Server startup script"""

import uvicorn
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    required = ['fastapi', 'uvicorn', 'openai', 'pinecone', 'httpx', 'sqlalchemy', 'psycopg2']
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {missing}")
        return False
    return True

def main():
    """Start the server"""
    if not check_dependencies():
        sys.exit(1)
    
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    try:
        uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()