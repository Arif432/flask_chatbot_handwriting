import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # Document Processing Configuration
    MEDICAL_DOCS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'medical_docs')
    
    # Embedding Store Configuration
    FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index')
    
    # Supported Diseases
    SUPPORTED_DISEASES = ['kidney', 'diabetes', 'heart']