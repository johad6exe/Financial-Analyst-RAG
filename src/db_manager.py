import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()   

# Get DB URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    """Creates the table if it doesn't exist."""
    if not DATABASE_URL:
        print("⚠️ No Database URL found. Chat history will not be saved.")
        return

    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_message(role, content):
    """Saves a message to the cloud DB."""
    if not DATABASE_URL: return
    
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (role, content) VALUES (%s, %s)",
        (role, content)
    )
    conn.commit()
    cur.close()
    conn.close()

def load_history(limit=10):
    """Loads the last N messages."""
    if not DATABASE_URL: return []

    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content FROM chat_history 
        ORDER BY timestamp ASC 
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    
    return [{"role": r[0], "content": r[1]} for r in rows]