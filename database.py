import sqlite3
from datetime import datetime

DB_NAME = "plates_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plate_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_plate(plate_number, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO plate_logs (plate_number, confidence, timestamp)
        VALUES (?, ?, ?)
    ''', (plate_number, confidence, now))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, plate_number, confidence, timestamp FROM plate_logs ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows
