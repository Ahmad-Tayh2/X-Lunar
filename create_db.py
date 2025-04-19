"""
Script to initialize the database for the faculty information chatbot using XAMPP MySQL.
"""

import pymysql
from database.db import init_db

def create_database_if_not_exists():
    """Create the database if it doesn't already exist."""
    # Database connection parameters for XAMPP
    DB_HOST = "localhost" 
    DB_USER = "root"
    DB_PASSWORD = ""  # Default XAMPP has no password for root
    DB_NAME = "faculty_chatbot"
    
    print("Connecting to MySQL server...")
    try:
        # Connect to MySQL server without specifying a database
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        print("Connected successfully to MySQL server")
        
        with conn.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
            print(f"Database '{DB_NAME}' created or already exists")
            
        conn.close()
        return True
    except Exception as e:
        print(f"Error connecting to MySQL server: {e}")
        print("Make sure XAMPP's MySQL service is running")
        return False

if __name__ == "__main__":
    print("Initializing database in XAMPP MySQL...")
    if create_database_if_not_exists():
        # Initialize the tables
        init_db()
        print("Database tables created successfully!")
    print("Database initialization complete!")