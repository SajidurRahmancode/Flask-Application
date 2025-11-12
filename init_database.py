
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_NAME = os.getenv('DB_NAME', 'flask_react_app')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    connection = None
    try:
        # Connect to MySQL server without specifying database
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{DB_NAME}' created or already exists")
            
        connection.commit()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False
    finally:
        if connection:
            connection.close()

def create_tables_if_not_exist():
    """Create tables if they don't exist"""
    connection = None
    try:
        # Connect to the specific database
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            # Create User table
            create_user_table = """
            CREATE TABLE IF NOT EXISTS `user` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `username` VARCHAR(80) NOT NULL UNIQUE,
                `email` VARCHAR(120) NOT NULL UNIQUE,
                `password_hash` VARCHAR(128) NOT NULL,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                `is_active` BOOLEAN DEFAULT TRUE,
                INDEX `idx_username` (`username`),
                INDEX `idx_email` (`email`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_user_table)
            print("✅ User table created or already exists")
            
            # Create Image table
            create_image_table = """
            CREATE TABLE IF NOT EXISTS `image` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `title` VARCHAR(200) NOT NULL,
                `url` VARCHAR(200) NOT NULL,
                `user_id` INT NOT NULL,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE,
                INDEX `idx_user_id` (`user_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_image_table)
            print("✅ Image table created or already exists")
            
        connection.commit()
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
    finally:
        if connection:
            connection.close()

def test_connection():
    """Test database connection and show info"""
    connection = None
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            # Get MySQL version
            cursor.execute("SELECT VERSION() as version")
            version = cursor.fetchone()
            
            # Get table count
            cursor.execute(f"SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = '{DB_NAME}'")
            table_info = cursor.fetchone()
            
            print(f"✅ Successfully connected to MySQL")
            print(f"📊 MySQL Version: {version['version']}")
            print(f"🏠 Host: {DB_HOST}:{DB_PORT}")
            print(f"💾 Database: {DB_NAME}")
            print(f"👤 User: {DB_USER}")
            print(f"📋 Tables in database: {table_info['table_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure XAMPP is running")
        print("2. Check MySQL service is started in XAMPP Control Panel")
        print("3. Verify database credentials in .env file")
        print("4. Ensure MySQL port 3306 is not blocked")
        return False
    finally:
        if connection:
            connection.close()

def initialize_database():
    """Main function to initialize the database"""
    print("🚀 Initializing MySQL database for Flask React App...")
    print("=" * 60)
    
    # Step 1: Create database
    if not create_database_if_not_exists():
        print("\n❌ Database initialization failed - could not create database")
        return False
    
    # Step 2: Create tables
    if not create_tables_if_not_exist():
        print("\n❌ Database initialization failed - could not create tables")
        return False
    
    # Step 3: Test connection
    if not test_connection():
        print("\n❌ Database initialization failed - connection test failed")
        return False
    
    print("\n✅ Database initialization completed successfully!")
    print("🔥 Database is ready for Flask app")
    return True

if __name__ == "__main__":
    success = initialize_database()
    if not success:
        exit(1)