import google.generativeai as genai

class RAGModel:
    def __init__(self, config, embedding_store):
        """
        Initialize the RAG model.
        
        Args:
            config (Config): Configuration object
            embedding_store (EmbeddingStore): Embedding store instance
        """
        self.config = config
        self.embedding_store = embedding_store
        
        # Configure Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.generation_model = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, query: str) -> str:
        """
        Generate a response using RAG approach.
        
        Args:
            query (str): User's query
        
        Returns:
            str: Generated response
        """
        # Retrieve relevant context
        context = self.embedding_store.search_embeddings(query)
        
        # Prepare prompt with context
        prompt = f"""You are a medical chatbot specialized in kidney, diabetes, and heart diseases.
        
        Context: {' '.join(context)}
        
        User Query: {query}
        
        Based on the context and your medical knowledge, provide a comprehensive and precise answer. 
        If the query is not related to kidney, diabetes, or heart diseases, politely inform the user.
        """
        
        # Generate response
        response = self.generation_model.generate_content(prompt)
        
        return response.text