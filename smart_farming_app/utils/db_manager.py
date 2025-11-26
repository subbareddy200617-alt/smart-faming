import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os

DB_NAME = 'agrisense_users.db'

def init_db():
    """Initializes the SQLite database and creates the users table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            user_role TEXT NOT NULL DEFAULT 'Farmer' -- Admin, Expert, FieldOfficer, Farmer
        )
    ''')
    conn.commit()
    conn.close()
    
    # Create initial admin and farmer users if they don't exist
    create_initial_users()

def create_initial_users():
    """Creates default users (admin and standard) if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Securely hash passwords
    admin_password_hash = generate_password_hash('adminpass')
    farmer_password_hash = generate_password_hash('farmerpass')
    expert_password_hash = generate_password_hash('expertpass')

    # Users to ensure exist
    initial_users = [
        ('admin', 'admin@agri.com', admin_password_hash, 'Admin'),
        ('FarmerDemo', 'farmer@agri.com', farmer_password_hash, 'Farmer'),
        ('ExpertUser', 'expert@agri.com', expert_password_hash, 'Expert'),
    ]

    for username, email, password_hash, role in initial_users:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is None:
            cursor.execute("INSERT INTO users (username, email, password_hash, user_role) VALUES (?, ?, ?, ?)",
                           (username, email, password_hash, role))
            print(f"User '{username}' created (Role: {role}).")
    
    conn.commit()
    conn.close()

def get_user_by_username(username):
    """Fetches user data from the database by username."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, user_role FROM users WHERE username = ?", (username,))
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        # Convert tuple result to a dictionary for easier access
        user = {
            'id': user_data[0],
            'username': user_data[1],
            'password_hash': user_data[2],
            'user_role': user_data[3]
        }
        return user
    return None

def verify_user(username, password):
    """Checks credentials against the stored hash."""
    user = get_user_by_username(username)
    if user and check_password_hash(user['password_hash'], password):
        return user
    return None

if __name__ == '__main__':
    # Initialize the DB if this script is run directly
    init_db()
    
    # Test Verification
    print(f"\nTest Logins:")
    print(f"Admin login success: {verify_user('admin', 'adminpass') is not None}")
    print(f"Farmer login success: {verify_user('FarmerDemo', 'farmerpass') is not None}")
    print(f"Bad login success: {verify_user('admin', 'wrongpass') is not None}")