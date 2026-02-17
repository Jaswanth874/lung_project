# Application Constants

# File size limits (in bytes)
FILE_SIZE_LIMITS = {
    'image': 2 * 1024 * 1024,  # 2 MB
    'document': 5 * 1024 * 1024,  # 5 MB
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': ['jpg', 'jpeg', 'png', 'gif'],
    'document': ['pdf', 'docx', 'txt'],
}

# Database settings
DATABASE_SETTINGS = {
    'host': 'localhost',
    'port': 5432,
    'username': 'user',
    'password': 'password',
    'db_name': 'lung_project_db',
}

# API configurations
API_CONFIG = {
    'BASE_URL': 'https://api.example.com',
    'TIMEOUT': 30,
    'RETRY_COUNT': 3,
}
