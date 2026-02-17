"""
LLM integration for RAG report generation using Google Gemini.
"""
import os
from typing import Optional, List, Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


def get_gemini_client() -> Optional[Any]:
    """
    Initialize and return Gemini client.
    
    Returns:
        Gemini client or None if not available
    """
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        except ImportError:
            pass
    
    if not api_key:
        print("Warning: Google API key not found. Set GOOGLE_API_KEY environment variable.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None


def generate_text(prompt: str, model_name: str = 'gemini-pro') -> Optional[str]:
    """
    Generate text using Gemini LLM.
    
    Args:
        prompt: Input prompt
        model_name: Model name to use
    
    Returns:
        Generated text or None if generation fails
    """
    client = get_gemini_client()
    
    if client is None:
        return None
    
    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Text generation failed: {e}")
        return None
