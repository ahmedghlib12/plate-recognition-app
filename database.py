import sqlite3
from datetime import datetime

DB_NAME = "plates_log.db"

def init_db():
    """إنشاء الجدول عند بداية التشغيل إذا لم يكن موجوداً"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plate_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_plate(plate_text, confidence):
    """إضافة لوحة جديدة مع الوقت والتاريخ"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO plate_logs (plate_text, confidence, timestamp)
        VALUES (?, ?, ?)
    ''', (plate_text, confidence, now))
    conn.commit()
    conn.close()

def get_all_logs():
    """استرجاع جميع السجلات المحفوظة"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, plate_text, confidence, timestamp FROM plate_logs ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows