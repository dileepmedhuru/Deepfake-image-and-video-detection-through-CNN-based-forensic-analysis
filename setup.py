"""
Setup Script for Forgery Detection System
Run this to initialize the project
"""

import os
import sys
import sqlite3
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        PROJECT_ROOT / 'database',
        PROJECT_ROOT / 'backend' / 'uploads' / 'images',
        PROJECT_ROOT / 'backend' / 'uploads' / 'videos',
        PROJECT_ROOT / 'ml_models',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory.relative_to(PROJECT_ROOT)}")
    
    return True

def init_database():
    """Initialize database with schema"""
    print("\n💾 Initializing database...")
    
    db_path = PROJECT_ROOT / 'database' / 'deepfake.db'
    schema_path = PROJECT_ROOT / 'database' / 'schema.sql'
    
    # Remove old database if exists
    if db_path.exists():
        print(f"   Removing old database...")
        db_path.unlink()
    
    # Check if schema exists
    if not schema_path.exists():
        print(f"   ❌ Error: Schema file not found at {schema_path}")
        return False
    
    # Create database
    print(f"   Creating database at: {db_path}")
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    
    with open(schema_path, 'r') as f:
        schema = f.read()
        cursor.executescript(schema)
    
    connection.commit()
    
    # Verify users
    cursor.execute("SELECT id, full_name, email, password, is_admin FROM users")
    users = cursor.fetchall()
    
    print(f"   ✓ Database created: {db_path.relative_to(PROJECT_ROOT)}")
    print(f"\n   Default accounts created:")
    for user_id, name, email, password, is_admin in users:
        role = "ADMIN" if is_admin else "USER"
        print(f"   • {role}: {email} / {password}")
    
    connection.close()
    
    # Verify database file exists and has correct permissions
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"\n   ✓ Database file created ({size} bytes)")
        print(f"   ✓ Database is writable: {os.access(str(db_path), os.W_OK)}")
        return True
    else:
        print(f"   ❌ Database file was not created!")
        return False

def verify_structure():
    """Verify directory structure"""
    print("\n🔍 Verifying project structure...")
    
    required_files = [
        'database/schema.sql',
        'backend/app.py',
        'backend/config.py',
        'backend/requirements.txt',
    ]
    
    required_dirs = [
        'frontend',
        'backend',
        'database',
    ]
    
    all_ok = True
    
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            all_ok = False
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"   ✓ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - MISSING!")
            all_ok = False
    
    return all_ok

def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8 or higher required!")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def setup():
    """Main setup function"""
    print("=" * 70)
    print("FORGERY DETECTION SYSTEM - SETUP")
    print("=" * 70)
    print(f"\nProject root: {PROJECT_ROOT}")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Verify structure
    if not verify_structure():
        print("\n⚠️  Some required files/folders are missing!")
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Initialize database
    if not init_database():
        return False
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("\n1. Install dependencies:")
    print("   cd backend")
    print("   pip install -r requirements.txt")
    print("\n2. Start the application:")
    print("   python app.py")
    print("\n3. Open in browser:")
    print("   http://localhost:5000")
    print("\n👤 Default Login Credentials:")
    print("   Admin: admin@deepfake.com / admin123")
    print("   User:  test@example.com / test123")
    print("\n⚠️  WARNING: Passwords stored in plain text (demo only!)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = setup()
        if not success:
            print("\n❌ Setup failed! Please check errors above.")
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)