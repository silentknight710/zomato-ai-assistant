# src/embedding_utils.py
"""
Utilities for loading the sentence transformer model (embedding model)
and generating vector embeddings for text chunks.
Includes simple caching for the loaded model.
"""
import time
import logging
from typing import Optional, List, Tuple, Any

# Attempt to import SentenceTransformer for type hinting
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformerModel = SentenceTransformer # Alias for clarity
except ImportError:
    logging.warning("Could not import SentenceTransformer type. Using 'Any'. Ensure 'sentence-transformers' is installed.")
    SentenceTransformerModel = Any

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Module-level cache for the model ---
_model: Optional[SentenceTransformerModel] = None
_model_name: Optional[str] = None
_embedding_dim: int = 0

# --- Function to load the model ---
def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2'
                        ) -> Tuple[Optional[SentenceTransformerModel], int]:
    """
    Loads and caches the Sentence Transformer model.

    Args:
        model_name (str): The name of the Hugging Face sentence-transformer model
                          (e.g., 'all-MiniLM-L6-v2').

    Returns:
        Tuple[Optional[SentenceTransformerModel], int]: A tuple containing the loaded
                                                       model object (or None if failed)
                                                       and the embedding dimension (0 if failed).
    """
    global _model, _model_name, _embedding_dim

    # Return cached model if it matches the requested name
    if _model is not None and _model_name == model_name:
        logger.debug(f"Using cached embedding model: {model_name}")
        return _model, _embedding_dim

    logger.info(f"Loading embedding model: {model_name}...")
    try:
        start_time = time.time()
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        # Attempt to get dimension, handle potential errors if model loading failed partially
        try:
            _embedding_dim = _model.get_sentence_embedding_dimension()
        except Exception as dim_err:
             logger.error(f"Could not get embedding dimension for model '{model_name}': {dim_err}")
             _embedding_dim = 0 # Set dimension to 0 if retrieval fails

        end_time = time.time()
        if _embedding_dim > 0:
             logger.info(f"Embedding model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
             logger.info(f"Embedding dimension: {_embedding_dim}")
        else:
             # If dimension is 0, loading likely had an issue even if no exception was raised before
             logger.error(f"Embedding model '{model_name}' loaded but failed to get valid dimension.")
             _model = None # Ensure model is None if loading wasn't fully successful
             _model_name = None

        return _model, _embedding_dim

    except ImportError:
         logger.exception("Error: 'sentence-transformers' library not found. Please install it.")
         _model, _model_name, _embedding_dim = None, None, 0
         return None, 0
    except Exception as e:
        logger.exception(f"Error loading embedding model '{model_name}': {e}")
        # Reset cache variables on any error
        _model, _model_name, _embedding_dim = None, None, 0
        return None, 0

# --- Function to generate embeddings ---
def generate_embeddings(model: Optional[SentenceTransformerModel],
                        text_list: List[str]
                       ) -> Optional[List[List[float]]]:
    """
    Generates vector embeddings for a list of text strings using the provided model.

    Args:
        model (Optional[SentenceTransformerModel]): The loaded Sentence Transformer model object.
        text_list (List[str]): A list of text strings to embed.

    Returns:
        Optional[List[List[float]]]: A list of vector embeddings (each embedding is a
                                     list of floats), or None if an error occurs.
    """
    if model is None:
        logger.error("Cannot generate embeddings: Embedding model is not loaded.")
        return None
    # Input type validation
    if not isinstance(text_list, list):
        logger.error(f"Input must be a list of text strings, got {type(text_list)}.")
        return None
    if not text_list: # Handle empty list case
         logger.warning("Input text list is empty. Returning empty list for embeddings.")
         return []

    logger.info(f"Generating embeddings for {len(text_list)} text chunk(s)...")
    try:
        start_time = time.time()
        # Use show_progress_bar=False for cleaner logs unless debugging large batches
        embeddings = model.encode(text_list, show_progress_bar=False).tolist() # Convert numpy array to list
        end_time = time.time()
        logger.info(f"Embeddings generated successfully in {end_time - start_time:.2f} seconds.")
        logger.debug(f"Generated {len(embeddings)} embedding vectors.")
        return embeddings
    except TypeError as te:
         # Catch potential errors if input list contains non-string elements
         logger.exception(f"TypeError during embedding generation. Ensure input list contains only strings: {te}")
         return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during embedding generation: {e}")
        return None