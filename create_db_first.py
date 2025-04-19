"""
Script to create the database first without trying to create tables.
"""

import pymysql

# Database connection variables
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "faculty_chatbot"

print("Connecting to MySQL server...")
try:
    # Connect to MySQL server without database
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
        
        # Show that we can use the database
        cursor.execute(f"USE {DB_NAME}")
        print(f"Successfully switched to database '{DB_NAME}'")
        
        # Show tables (should be empty at first)
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"Current tables: {tables}")
        
    conn.close()
    print("Database setup complete!")
    
except Exception as e:
    print(f"Error: {e}")
    print("MySQL connection details:")
    print(f"Host: {DB_HOST}, User: {DB_USER}, Password: {'<empty>' if DB_PASSWORD == '' else '<set>'}")