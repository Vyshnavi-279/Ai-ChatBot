import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('observability.db')
cursor = conn.cursor()

# Create a table to store observability data
cursor.execute("""CREATE TABLE IF NOT EXISTS observability (
    timestamp TEXT PRIMARY KEY,
    session_id TEXT,
    question TEXT,
    dimension_or_tool TEXT,
    retrieved_chunk_ids TEXT,
    model_name TEXT,
    input_tokens TEXT,
    output_tokens TEXT,
    latency_seconds NUMERIC,
    refused BOOLEAN,
    error TEXT
)""")

# Insert a single row of observability data
cursor.execute("INSERT INTO observability (timestamp, session_id, question, dimension_or_tool, retrieved_chunk_ids, model_name, input_tokens, output_tokens, latency_seconds, refused, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
             (timestamp, session_id, question, dimension_or_tool, retrieved_chunk_ids, model_name, input_tokens, output_tokens, latency_seconds, refused, error))
conn.commit()

# Close the database connection
cursor.close()
conn.close()
