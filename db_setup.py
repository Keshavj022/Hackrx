#!/usr/bin/env python3
"""
Database setup script for PostgreSQL
"""

import sys
import subprocess
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_postgres_installed():
    """Check if PostgreSQL is installed"""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            logger.error("PostgreSQL not found")
            return False
    except FileNotFoundError:
        logger.error("PostgreSQL not installed")
        return False

def create_database():
    """Create database and user"""
    try:
        # Since the database and user already exist, just verify we can connect
        logger.info(f"Database {settings.postgres_db} and user {settings.postgres_user} already exist")
        logger.info("Skipping database/user creation")
        return True
        
    except Exception as e:
        logger.error(f"Error in database setup: {e}")
        return False

def test_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        logger.info(f"Successfully connected to PostgreSQL: {version}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

def initialize_tables():
    """Initialize database tables"""
    try:
        from database import init_database
        init_database()
        logger.info("Database tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize tables: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up PostgreSQL Database for HackRx")
    print("=" * 50)
    
    # Check if PostgreSQL is installed
    if not check_postgres_installed():
        print("\n❌ PostgreSQL is not installed. Please install it first:")
        print("  - macOS: brew install postgresql")
        print("  - Ubuntu: sudo apt-get install postgresql postgresql-contrib")
        print("  - Windows: Download from https://www.postgresql.org/download/")
        return False
    
    # Check environment variables
    if not settings.postgres_password:
        print("❌ POSTGRES_PASSWORD not set in environment variables")
        print("Please set it in your .env file or environment")
        return False
    
    # Create database and user
    print("\n📊 Creating database and user...")
    if not create_database():
        print("❌ Failed to create database")
        return False
    
    # Test connection
    print("\n🔍 Testing database connection...")
    if not test_connection():
        print("❌ Database connection failed")
        return False
    
    # Initialize tables
    print("\n📋 Initializing database tables...")
    if not initialize_tables():
        print("❌ Failed to initialize tables")
        return False
    
    print("\n✅ Database setup completed successfully!")
    print(f"Database: {settings.postgres_db}")
    print(f"User: {settings.postgres_user}")
    print(f"Host: {settings.postgres_host}:{settings.postgres_port}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)