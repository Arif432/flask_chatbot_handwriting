import os
import faiss
import numpy as np
from typing import List, Dict
import json
import google.generativeai as genai

class EmbeddingStore:
    def __init__(self, config):
        """
        Initialize the embedding store.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        
        # Configure Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.embedding_model = genai.embed_content
        
        # Create index directory if not exists
        os.makedirs(self.config.FAISS_INDEX_PATH, exist_ok=True)
    
    def create_embeddings(self, texts: Dict[str, str]) -> None:
        """
        Create and save FAISS index for the given texts.
        
        Args:
            texts (Dict[str, str]): Dictionary of disease texts
        """
        all_embeddings = []
        all_texts = []
        disease_mapping = {}
        
        for disease, text in texts.items():
            # Split text into chunks
            chunks = self._split_text(text)
            
            # Generate embeddings for chunks
            disease_embeddings = []
            for chunk in chunks:
                embedding = self._get_embedding(chunk)
                disease_embeddings.append(embedding)
                all_embeddings.append(embedding)
                all_texts.append(chunk)
            
            # Track mapping of disease to text chunks
            disease_mapping[disease] = len(disease_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save index and metadata
        faiss.write_index(index, os.path.join(self.config.FAISS_INDEX_PATH, 'medical_index.faiss'))
        
        # Optional: Save additional metadata
        import json
        with open(os.path.join(self.config.FAISS_INDEX_PATH, 'metadata.json'), 'w') as f:
            json.dump({
                'texts': all_texts,
                'disease_mapping': disease_mapping
            }, f)
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text (str): Input text
            chunk_size (int): Size of each text chunk
        
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text chunk using Gemini.
        
        Args:
            text (str): Input text chunk
        
        Returns:
            np.ndarray: Embedding vector
        """
        result = self.embedding_model(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])
    
    def search_embeddings(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search embeddings for the most relevant text chunks.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            List[str]: Most relevant text chunks
        """
        # Load index and metadata
        index = faiss.read_index(os.path.join(self.config.FAISS_INDEX_PATH, 'medical_index.faiss'))
        
        with open(os.path.join(self.config.FAISS_INDEX_PATH, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Get query embedding
        query_embedding = self._get_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        distances, indices = index.search(query_embedding, top_k)
        
        # Retrieve relevant texts
        relevant_texts = [metadata['texts'][i] for i in indices[0]]
        
        return relevant_texts