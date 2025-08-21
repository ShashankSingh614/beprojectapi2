from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from groq import Groq
import re
import logging
from typing import Dict, List, Optional, Union
import time
import os
from pathlib import Path
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "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"  # Hardcoded API key
FILE_PATH = Path(os.getenv("BNS_DATASET_PATH", "dataset/bnsdataset.xlsx")).expanduser()
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L12-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))
TOP_K_MATCHES = int(os.getenv("TOP_K_MATCHES", 3))

class QueryRequest(BaseModel):
    query: str
    include_alternatives: bool = False
    similarity_threshold: float = SIMILARITY_THRESHOLD

class BNSSearchSystem:
    def __init__(self, file_path: str, api_key: str):
        self.file_path = file_path
        self.api_key = api_key
        self.dataset = None
        self.model = None
        self.embeddings = None
        self.client = None
        self.required_columns = [
            'Section_Number', 'Subsection_Number', 'Title', 'Content', 
            'Explanation', 'Exception', 'Illustrations', 'Punishment', 'Cross_References'
        ]
        
        # Initialize system
        self._load_dataset()
        self._initialize_model()
        self._initialize_groq_client()
        self._create_embeddings()
    
    def _load_dataset(self):
        """Load and validate dataset"""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"Dataset file not found: {self.file_path}")
                raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
            
            logger.info(f"Loading dataset from {self.file_path}")
            self.dataset = pd.read_excel(self.file_path)
            
            # Handle missing values
            self.dataset = self.dataset.fillna({
                'Content': 'No content available',
                'Explanation': 'No explanation available',
                'Exception': 'No exceptions mentioned',
                'Illustrations': 'No illustrations provided',
                'Punishment': 'No punishment specified',
                'Cross_References': 'No cross references',
                'Title': 'Untitled section'
            })
            
            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in self.dataset.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Dataset loaded successfully with {len(self.dataset)} rows")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _initialize_groq_client(self):
        """Initialize Groq client"""
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
            raise
    
    def _create_embeddings(self):
        """Create embeddings for all content"""
        try:
            logger.info("Creating embeddings for dataset...")
            
            # Combine multiple fields for better matching
            combined_text = []
            for _, row in self.dataset.iterrows():
                text_parts = [
                    str(row['Title']),
                    str(row['Content']),
                    str(row['Explanation']),
                    str(row['Illustrations'])
                ]
                combined = " ".join([part for part in text_parts if part and part != 'nan'])
                combined_text.append(combined)
            
            self.embeddings = self.model.encode(combined_text, show_progress_bar=True)
            logger.info(f"Created embeddings for {len(combined_text)} entries")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _extract_section_number(self, query: str) -> Optional[str]:
        """Extract section number from query if present"""
        patterns = [
            r'section\s+(\d+[a-zA-Z]?)',
            r'sec\s+(\d+[a-zA-Z]?)',
            r'§\s*(\d+[a-zA-Z]?)',
            r'\b(\d{1,3}[a-zA-Z]?)\b\s*(?:of\s*BNS)?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
        return None
    
    def _search_by_section_number(self, section_num: str) -> Optional[pd.Series]:
        """Search for exact section number match"""
        try:
            # Handle alphanumeric section numbers (e.g., "123A")
            matches = self.dataset[self.dataset['Section_Number'].astype(str).str.lower() == section_num.lower()]
            if not matches.empty:
                return matches.iloc[0]
        except (ValueError, IndexError):
            pass
        return None
    
    def _convert_numpy(self, obj):
        """Convert numpy objects to native Python types"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    @lru_cache(maxsize=100)
    def _generate_response_cached(self, title: str, content: str, explanation: str, 
                                 exception: str, illustrations: str, punishment: str) -> str:
        """Cached version of response generation to avoid repeated API calls"""
        return self._generate_human_like_response_internal(
            title, content, explanation, exception, illustrations, punishment
        )
    
    def _generate_human_like_response_internal(self, title: str, content: str, explanation: str,
                                             exception: str, illustrations: str, punishment: str) -> str:
        """Generate human-like response using Groq API with fallback"""
        prompt = f"""Based on the following legal information from the Bharatiya Nyaya Sanhita (BNS), create a comprehensive and clear explanation for the general public:

Title: {title}
Content: {content}
Explanation: {explanation}
Exception: {exception}
Illustrations: {illustrations}
Punishment: {punishment}

Instructions:
1. Start with "According to the Bharatiya Nyaya Sanhita (BNS),"
2. Explain the law in simple, understandable terms
3. Include any exceptions clearly
4. Integrate illustrations naturally into the explanation
5. Mention the punishment if applicable
6. Keep the tone informative but accessible
7. Make it flow as a cohesive paragraph

Please provide a detailed, human-readable explanation:"""

        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a legal expert who explains Indian laws in simple, clear language for the general public."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=True,
                stop=None
            )
            
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with Groq API: {e}")
            # Fallback response
            return (f"According to the Bharatiya Nyaya Sanhita (BNS), {title}: {content}. "
                    f"{explanation} {exception} {illustrations} Punishment: {punishment}")
    
    def search(self, user_query: str, include_alternatives: bool = False, 
               similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        """Enhanced search with multiple matching strategies"""
        try:
            if not user_query or not user_query.strip():
                return {
                    "status": "error",
                    "message": "Query cannot be empty"
                }
            
            logger.info(f"Processing query: {user_query}, include_alternatives={include_alternatives}, threshold={similarity_threshold}")
            
            # Strategy 1: Try exact section number match first
            section_num = self._extract_section_number(user_query)
            if section_num:
                exact_match = self._search_by_section_number(section_num)
                if exact_match is not None:
                    logger.info(f"Found exact section match: {section_num}")
                    return self._format_response(exact_match, "Exact Section Match")
            
            # Strategy 2: Semantic similarity search
            query_embedding = self.model.encode(user_query)
            similarities = util.cos_sim(query_embedding, self.embeddings)[0].numpy()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:TOP_K_MATCHES]
            top_similarities = similarities[top_indices]
            
            # Check if best match meets threshold
            best_similarity = top_similarities[0]
            if best_similarity < similarity_threshold:
                return {
                    "status": "no_match",
                    "message": f"No relevant sections found with similarity > {similarity_threshold:.2f}. Best match similarity: {best_similarity:.2f}",
                    "suggestion": "Try rephrasing your query or using more specific legal terms."
                }
            
            # Get best match
            best_match_idx = top_indices[0]
            matched_row = self.dataset.iloc[best_match_idx]
            
            response = self._format_response(matched_row, "Semantic Match", best_similarity)
            
            # Add alternative matches if requested
            if include_alternatives and len(top_indices) > 1:
                alternatives = []
                for i in range(1, min(len(top_indices), 3)):
                    if top_similarities[i] >= similarity_threshold:
                        alt_row = self.dataset.iloc[top_indices[i]]
                        alternatives.append({
                            "section_number": self._convert_numpy(alt_row['Section_Number']),
                            "title": alt_row['Title'],
                            "similarity_score": float(top_similarities[i])
                        })
                
                if alternatives:
                    response["alternatives"] = alternatives
            
            return response
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return {
                "status": "error",
                "message": f"An error occurred while processing your query: {str(e)}"
            }
    
    def _format_response(self, matched_row: pd.Series, match_type: str, 
                        similarity_score: float = None) -> Dict:
        """Format the response in a consistent structure"""
        try:
            # Generate human-like explanation
            human_explanation = self._generate_response_cached(
                str(matched_row['Title']),
                str(matched_row['Content']),
                str(matched_row['Explanation']),
                str(matched_row['Exception']),
                str(matched_row['Illustrations']),
                str(matched_row['Punishment'])
            )
            
            response = {
                "status": "success",
                "match_type": match_type,
                "category": "Criminal Law",
                "section_number": self._convert_numpy(matched_row['Section_Number']),
                "subsection_number": self._convert_numpy(matched_row['Subsection_Number']),
                "title": matched_row['Title'],
                "content": matched_row['Content'],
                "explanation": human_explanation,
                "original_explanation": matched_row['Explanation'],
                "exception": matched_row['Exception'],
                "illustrations": matched_row['Illustrations'],
                "punishment": matched_row['Punishment'],
                "cross_references": matched_row['Cross_References']
            }
            
            if similarity_score is not None:
                response["similarity_score"] = float(similarity_score)
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                "status": "error",
                "message": f"Error formatting response: {str(e)}"
            }

# Initialize the system
try:
    bns_system = BNSSearchSystem(FILE_PATH, API_KEY)
    logger.info("BNS Search System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize BNS Search System: {e}")
    bns_system = None

def modelRun(user_query: str, include_alternatives: bool = False, 
             similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict:
    """Main function to run the model with improved features"""
    if not user_query or not user_query.strip():
        return {
            "status": "error",
            "message": "Query cannot be empty"
        }
    if bns_system is None:
        return {
            "status": "error",
            "message": "BNS Search System not initialized. Please check the dataset file and API key."
        }
    
    return bns_system.search(user_query, include_alternatives, similarity_threshold)