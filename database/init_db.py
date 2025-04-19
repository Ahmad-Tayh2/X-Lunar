"""
Script to initialize the database for the faculty information chatbot.
"""

import os
import pymysql
from database.db import init_db

def create_database_if_not_exists():
    """Create the database if it doesn't already exist."""
    # Parse connection string to get database name
    connection_string = os.getenv(
        "DB_CONNECTION_STRING", 
        "mysql+pymysql://root@localhost:3306/faculty_chatbot"
    )
    
    # Extract host, user, password and db name from connection string
    parts = connection_string.split("://")[1].split("@")
    user_password = parts[0].split(":")
    username = user_password[0]
    
    host_db = parts[1].split("/")
    host = host_db[0]
    db_name = host_db[1]
    
    # Connect to MySQL server without specifying a database
    conn = pymysql.connect(
        host=host,
        user=username,
        password=""  # No password
    )
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            print(f"Database '{db_name}' created or already exists")
    finally:
        conn.close()

if __name__ == "__main__":
    print("Initializing database...")
    create_database_if_not_exists()
    init_db()
    print("Database initialization complete!")