"""
WSGI entry point for production deployment.
Used by Gunicorn and other WSGI servers.
"""
from app import app

if __name__ == "__main__":
    app.run()
