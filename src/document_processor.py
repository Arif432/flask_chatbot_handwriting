import os
import PyPDF2
from typing import List, Dict

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdfs(folder_path: str) -> Dict[str, str]:
        """
        Extract text from PDF files in the specified folder.
        
        Args:
            folder_path (str): Path to the folder containing medical PDFs
        
        Returns:
            Dict[str, str]: Dictionary of filename to extracted text
        """
        extracted_texts = {}
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                filepath = os.path.join(folder_path, filename)
                
                try:
                    with open(filepath, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ''
                        for page in reader.pages:
                            text += page.extract_text() + '\n'
                        
                        # Use filename (without extension) as the key
                        disease_name = os.path.splitext(filename)[0]
                        extracted_texts[disease_name] = text
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return extracted_texts
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the extracted text.
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text