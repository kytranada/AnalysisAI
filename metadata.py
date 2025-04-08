import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
# Path to store column metadata
METADATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "column_metadata")
# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Similarity threshold for column matching
SIMILARITY_THRESHOLD = 0.7

class ColumnMetadataManager:
    """
    Manages column metadata extraction, embedding generation, storage, and similarity search.
    This class handles the mapping between user query terms and actual column names.
    """
    
    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the column metadata manager with the specified embedding model."""
        # Create metadata directory if it doesn't exist
        os.makedirs(METADATA_DIR, exist_ok=True)
        
        # Initialize the embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded successfully.")
        
        # Dictionary to store metadata for each dataset
        self.dataset_metadata = {}
        
    def extract_column_metadata(self, df: pd.DataFrame, file_path: str, llm) -> Dict[str, Dict]:
        """
        Extract metadata for each column in the DataFrame.
        Uses the LLM to generate descriptions and possible synonyms for each column.
        """
        dataset_id = os.path.basename(file_path)
        print(f"\nExtracting column metadata for dataset: {dataset_id}")
        
        # Initialize metadata dictionary for this dataset
        metadata = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "columns": {},
            "column_embeddings": {}
        }
        
        # Get basic statistics for each column
        stats = df.describe(include='all').to_dict()
        
        # Process each column
        for col_name in df.columns:
            print(f"Processing column: {col_name}")
            
            # Get column data type and sample values
            col_type = str(df[col_name].dtype)
            sample_values = df[col_name].dropna().head(5).tolist()
            
            # Get column statistics if available
            col_stats = {}
            if col_name in stats:
                col_stats = stats[col_name]
                # Convert numpy types to Python native types for JSON serialization
                col_stats = {k: v.item() if hasattr(v, 'item') else v 
                            for k, v in col_stats.items() if not pd.isna(v)}
            
            # Use LLM to generate column description and synonyms
            description, synonyms = self._generate_column_description(col_name, col_type, sample_values, llm)
            
            # Store column metadata
            metadata["columns"][col_name] = {
                "name": col_name,
                "type": col_type,
                "description": description,
                "synonyms": synonyms,
                "sample_values": [str(v) for v in sample_values],
                "statistics": col_stats
            }
            
            # Generate and store embeddings for column name and synonyms
            self._generate_column_embeddings(col_name, description, synonyms, metadata)
        
        # Save metadata to file
        self._save_metadata(dataset_id, metadata)
        
        # Store metadata in memory
        self.dataset_metadata[dataset_id] = metadata
        
        print(f"Column metadata extraction completed for {len(df.columns)} columns.")
        return metadata
    
    def _generate_column_description(self, col_name: str, col_type: str, sample_values: List, 
                                    llm) -> Tuple[str, List[str]]:
        """Use the LLM to generate a description and synonyms for the column."""
        prompt = f"""
        Analyze the following column from a dataset:
        - Column Name: {col_name}
        - Data Type: {col_type}
        - Sample Values: {sample_values}

        Based *only* on the information provided (column name, data type, sample values), perform the following tasks:

        1.  **Description:** Write a concise (1-2 sentences) description of the likely meaning or purpose of this column. Focus on what the data represents *based on the clues given*.
        2.  **Synonyms:** Generate a list of 3-5 relevant alternative names, abbreviations, or common phrasings that a user might realistically use to refer to *this specific column* in a natural language query.

        **Guidelines for Synonyms:**
        *   Think about how someone might naturally ask about this column.
        *   Consider variations in capitalization, spacing, or common abbreviations (e.g., 'first name' for 'first_name', 'ID' for 'identifier', 'qty' for 'quantity').
        *   Ensure synonyms are directly related to the column's inferred meaning from its name, type, and sample values.
        *   **IMPORTANT: AVOID generating:**
            *   Overly generic terms like 'value', 'data', 'column', 'field', 'information'.
            *   SQL keywords or function names (e.g., 'select', 'where', 'count', 'average', 'max', 'sum').
            *   Synonyms that are too similar to each other (provide distinct alternatives).
            *   Terms unlikely to be used in a query context.

        **Output Format:** Respond *only* with the following exact structure, replacing the bracketed text. Do not add any explanations or introductory text:
        Description: [Your concise description here]
        Synonyms: [synonym1, synonym2, synonym3, ...]
        """
        
        try:
            response = llm.invoke(prompt).content
            
            # Parse the response
            description_line = [line for line in response.split('\n') if line.startswith('Description:')]
            synonyms_line = [line for line in response.split('\n') if line.startswith('Synonyms:')]
            
            description = description_line[0].replace('Description:', '').strip() if description_line else f"Column containing {col_type} data"
            
            synonyms_text = synonyms_line[0].replace('Synonyms:', '').strip() if synonyms_line else ""
            synonyms = [s.strip() for s in synonyms_text.strip('[]').split(',')]
            synonyms = [s for s in synonyms if s]  # Remove empty strings
            
            return description, synonyms
        except Exception as e:
            print(f"Error generating column description: {e}")
            return f"Column containing {col_type} data", [col_name.replace('_', ' ')]
    
    def _generate_column_embeddings(self, col_name: str, description: str, synonyms: List[str], 
                                   metadata: Dict) -> None:
        """Generate embeddings for column name, description, and synonyms."""
        # Create text representations to embed
        texts_to_embed = [
            col_name,
            col_name.replace('_', ' '),  # Replace underscores with spaces
            description
        ] + synonyms
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts_to_embed)
        
        # Store embeddings
        metadata["column_embeddings"][col_name] = embeddings.tolist()
    
    def _save_metadata(self, dataset_id: str, metadata: Dict) -> None:
        """Save metadata to a JSON file."""
        file_path = os.path.join(METADATA_DIR, f"{dataset_id}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metadata = json.loads(json.dumps(metadata, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
        
        with open(file_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        print(f"Metadata saved to {file_path}")
    
    def load_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Load metadata from a JSON file."""
        file_path = os.path.join(METADATA_DIR, f"{dataset_id}.json")
        
        if not os.path.exists(file_path):
            print(f"Metadata file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            # Store metadata in memory
            self.dataset_metadata[dataset_id] = metadata
            
            print(f"Metadata loaded from {file_path}")
            return metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def find_matching_columns(self, query_terms: List[str], dataset_id: str) -> Dict[str, float]:
        """
        Find columns that match the query terms using embedding similarity.
        Returns a dictionary mapping column names to similarity scores.
        """
        # Ensure metadata is loaded
        if dataset_id not in self.dataset_metadata:
            metadata = self.load_metadata(dataset_id)
            if not metadata:
                print(f"No metadata available for dataset: {dataset_id}")
                return {}
        
        metadata = self.dataset_metadata[dataset_id]
        
        # Generate embeddings for query terms
        query_embeddings = self.embedding_model.encode(query_terms)
        
        # Find matching columns
        matches = {}
        for col_name, embeddings_list in metadata["column_embeddings"].items():
            # Convert stored embeddings back to numpy arrays
            col_embeddings = np.array(embeddings_list)
            
            # Calculate similarity scores
            max_similarity = 0
            for query_embedding in query_embeddings:
                for col_embedding in col_embeddings:
                    similarity = cosine_similarity([query_embedding], [col_embedding])[0][0]
                    max_similarity = max(max_similarity, similarity)
            
            # Store match if similarity is above threshold
            if max_similarity >= SIMILARITY_THRESHOLD:
                matches[col_name] = float(max_similarity)
        
        # Sort matches by similarity score (descending)
        matches = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
        
        return matches
    
    def get_column_info(self, col_name: str, dataset_id: str) -> Optional[Dict]:
        """Get information about a specific column."""
        if dataset_id not in self.dataset_metadata:
            metadata = self.load_metadata(dataset_id)
            if not metadata:
                return None
        
        metadata = self.dataset_metadata[dataset_id]
        
        if col_name in metadata["columns"]:
            return metadata["columns"][col_name]
        
        return None
    
    def preprocess_query(self, query: str, dataset_id: str) -> str:
        """
        Preprocess a user query by replacing column references with actual column names.
        This handles misspellings and alternative terms for columns.
        
        Improvements:
        - Considers multi-word phrases (n-grams) for better matching
        - More conservative with replacements (higher confidence threshold)
        - Avoids replacing terms inside quotes (string literals)
        - Avoids replacing SQL keywords and common terms
        """
        # Ensure metadata is loaded
        if dataset_id not in self.dataset_metadata:
            metadata = self.load_metadata(dataset_id)
            if not metadata:
                print(f"No metadata available for dataset: {dataset_id}")
                return query
        
        # Original query for logging
        original_query = query
        
        # SQL keywords and common terms to avoid replacing
        sql_keywords = {
            'select', 'from', 'where', 'group', 'order', 'by', 'having', 'and', 'or', 
            'not', 'null', 'is', 'as', 'in', 'join', 'left', 'right', 'inner', 'outer',
            'on', 'between', 'like', 'case', 'when', 'then', 'else', 'end', 'distinct',
            'count', 'sum', 'avg', 'min', 'max', 'limit', 'offset', 'union', 'except',
            'intersect', 'all', 'any', 'some', 'exists', 'true', 'false', 'asc', 'desc'
        }
        
        # Extract words and potential multi-word phrases (n-grams)
        words = query.lower().split()
        
        # Generate n-grams (1, 2, and 3-word phrases)
        ngrams = []
        # Add individual words
        ngrams.extend(words)
        # Add 2-word phrases
        if len(words) >= 2:
            ngrams.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
        # Add 3-word phrases
        if len(words) >= 3:
            ngrams.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
        
        # Sort n-grams by length (descending) to prioritize longer matches
        ngrams.sort(key=len, reverse=True)
        
        # Find matching columns for each n-gram
        replacements = {}
        for ngram in ngrams:
            # Skip short terms, SQL keywords, and common words
            if len(ngram) < 3 or ngram in sql_keywords:
                continue
            
            # Find matching columns
            matches = self.find_matching_columns([ngram], dataset_id)
            
            # Only consider high-confidence matches (using a higher threshold)
            high_confidence_matches = {k: v for k, v in matches.items() if v >= 0.8}
            
            if high_confidence_matches:
                # Get the best match
                best_match = list(high_confidence_matches.keys())[0]
                best_score = list(high_confidence_matches.values())[0]
                
                # Store the replacement with its score and the original term
                replacements[ngram] = {
                    'column': best_match,
                    'score': best_score,
                    'original': ngram
                }
        
        # Sort replacements by score (descending)
        sorted_replacements = sorted(
            replacements.items(), 
            key=lambda x: (x[1]['score'], len(x[0])), 
            reverse=True
        )
        
        # Process the query to avoid replacing terms in string literals
        # Find all string literals (text between quotes)
        string_literals = []
        quote_positions = []
        
        # Find positions of quotes (both single and double)
        for i, char in enumerate(query):
            if char in ["'", '"']:
                quote_positions.append(i)
        
        # Determine which parts of the query are inside string literals
        inside_quotes = False
        for i in range(len(query)):
            if i in quote_positions:
                inside_quotes = not inside_quotes
            if inside_quotes:
                string_literals.append(i)
        
        # Apply replacements, being careful not to modify string literals
        processed_query = query
        replacements_applied = []
        
        for term, replacement_info in sorted_replacements:
            column = replacement_info['column']
            
            # Skip if this term overlaps with already applied replacements
            skip = False
            for applied_term in replacements_applied:
                if term in applied_term or applied_term in term:
                    skip = True
                    break
            if skip:
                continue
            
            # Find all occurrences of the term in the query
            term_lower = term.lower()
            query_lower = processed_query.lower()
            
            start_pos = 0
            while start_pos < len(query_lower):
                pos = query_lower.find(term_lower, start_pos)
                if pos == -1:
                    break
                
                # Check if this occurrence is inside a string literal
                in_literal = False
                for i in range(pos, pos + len(term)):
                    if i in string_literals:
                        in_literal = True
                        break
                
                # Check if this is a standalone term (not part of another word)
                is_standalone = True
                if pos > 0 and query_lower[pos-1].isalnum():
                    is_standalone = False
                if pos + len(term) < len(query_lower) and query_lower[pos + len(term)].isalnum():
                    is_standalone = False
                
                # Only replace if not in a string literal and is a standalone term
                if not in_literal and is_standalone:
                    # Replace this specific occurrence
                    before = processed_query[:pos]
                    after = processed_query[pos + len(term):]
                    processed_query = before + column + after
                    
                    # Update the lowercase version for future searches
                    query_lower = processed_query.lower()
                    
                    # Add to applied replacements
                    replacements_applied.append(term)
                    
                    # Log the replacement
                    print(f"Replaced '{term}' with column '{column}' (confidence: {replacement_info['score']:.2f})")
                    
                    # Start next search after this replacement
                    start_pos = pos + len(column)
                else:
                    # Skip this occurrence
                    start_pos = pos + len(term)
        
        # Log if any changes were made
        if processed_query != original_query:
            print(f"Query preprocessed: '{original_query}' -> '{processed_query}'")
        
        return processed_query
