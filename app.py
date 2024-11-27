import pickle
from flask import Flask, render_template, request, url_for, jsonify, session
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter
from bson import ObjectId
import datetime
from ultralytics import YOLO
from flask_cors import CORS
from chatbot_service import get_chatbot_response,ask_gemini 
from pymongo import MongoClient
from datetime import datetime, timezone
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.document_processor import DocumentProcessor
from src.embedding_store import EmbeddingStore
from src.rag_model import RAGModel

from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from .env file
python_warnings = os.getenv('PYTHONWARNINGS')
kmp_duplicate_lib_ok = os.getenv('KMP_DUPLICATE_LIB_OK')

app = Flask(__name__)

config = Config()
document_processor = DocumentProcessor()
embedding_store = EmbeddingStore(config)
rag_model = RAGModel(config, embedding_store)

@app.route('/initialize', methods=['POST'])
def initialize_chatbot():
    """
    Initialize the chatbot by processing documents and creating embeddings.
    """
    try:
        # Extract texts from PDFs
        medical_texts = document_processor.extract_text_from_pdfs(config.MEDICAL_DOCS_FOLDER)
        
        # Create embeddings
        embedding_store.create_embeddings(medical_texts)
        
        return jsonify({
            'status': 'success', 
            'message': 'Chatbot initialized successfully',
            'processed_documents': list(medical_texts.keys())
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/chatRAG', methods=['POST'])
def chatRAG():
    try:
        # Detailed logging
        app.logger.info("Received request to /chatRAG")
        
        # More robust JSON parsing
        if not request.is_json:
            app.logger.error("Request content type is not JSON")
            return jsonify({
                'error': 'Invalid content type. Must be application/json',
                'details': str(request.content_type)
            }), 400

        data = request.get_json(force=True)
        
        # Comprehensive input validation
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({'error': 'Empty JSON payload'}), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            app.logger.error("Empty query received")
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Log incoming query
        app.logger.info(f"Processing query: {query}")
        
        # Generate response
        response = rag_model.generate_response(query)
        
        # Log response generation
        app.logger.info("Response generated successfully")
        
        return jsonify({
            'query': query,
            'response': response
        }), 200
    
    except Exception as e:
        # Comprehensive error logging
        app.logger.error(f"Error in chatRAG: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # More robust server configuration
    app.run(
        host='0.0.0.0',  # Listen on all available interfaces
        port=5000,
        debug=True,      # Detailed error messages
        threaded=True    # Handle multiple requests
    )
