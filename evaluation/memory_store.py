import sqlite3

memory_db_path = os.path.join(os.getcwd(), "memory.db")

conn = sqlite3.connect(memory_db_path)
cursor = conn.cursor()

class Session(sqlite3.Model):
    session_id = sqlite3.Column(primary_key=True)
    created_at = sqlite3.Column(nullable=True)
    summary_text = sqlite3.Column(nullable=True)
    updated_at = sqlite3.Column(nullable=True)

    def save_summary(self, session_id, summary_text):
        cursor.execute(
            "INSERT INTO sessions (session_id, created_at, summary_text, updated_at) VALUES (?, ?, ?, ?"
        )
        conn.commit()

    def should_summarize(self, turn_count):
        return turn_count <= 6

