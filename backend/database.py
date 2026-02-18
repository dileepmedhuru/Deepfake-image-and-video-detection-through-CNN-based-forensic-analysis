from flask_sqlalchemy import SQLAlchemy
import sqlite3
import os

db = SQLAlchemy()

def init_db(db_path='database/deepfake.db', schema_path='database/schema.sql'):
    """Initialize database with schema"""
    
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect and execute schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(schema_path, 'r') as f:
        schema = f.read()
        cursor.executescript(schema)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized successfully at {db_path}")