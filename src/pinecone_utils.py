# src/pinecone_utils.py
"""
Utilities for interacting with the Pinecone vector database.
Handles initialization, index creation, upserting, querying, and deletion.
Uses logging for status updates and errors.
"""
import time
import math
import logging
from typing import List, Dict, Optional, Any, Union

# Attempt to import specific Pinecone types for hinting, fallback to Any
try:
    from pinecone import Pinecone
    from pinecone.models import ServerlessSpec, PodSpec, IndexDescription, IndexList # Added IndexDescription, IndexList
    from pinecone.exceptions import ApiException # Import specific exception
    PineconeClient = Pinecone # Alias for clarity
    PineconeIndex = Any # The Index object type is complex, use Any for now
except ImportError:
    logging.warning("Could not import specific Pinecone types. Using 'Any'. Ensure 'pinecone' library is installed.")
    ApiException = Exception # Fallback exception
    PineconeClient = Any
    PineconeIndex = Any
    ServerlessSpec = Any
    IndexDescription = Any
    IndexList = Any

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Module-level cache for client and index connection ---
_pinecone_client: Optional[PineconeClient] = None
_pinecone_index: Optional[PineconeIndex] = None
_index_name_cache: Optional[str] = None

# --- Initialization Function ---
def init_pinecone(api_key: Optional[str],
                  index_name: str,
                  dimension: int,
                  spec_config: Optional[Dict[str, str]] = None
                 ) -> Optional[PineconeIndex]:
    """
    Initializes Pinecone connection and connects to the specified index.
    Creates the index if it doesn't exist using ServerlessSpec.
    Caches the connection and index object for reuse.

    Args:
        api_key (Optional[str]): Your Pinecone API key. Reads from environment if None.
        index_name (str): The name of the Pinecone index.
        dimension (int): The dimension of the vectors for the index.
        spec_config (Optional[Dict[str, str]]): Configuration for ServerlessSpec
                                                (cloud, region), required for index creation.

    Returns:
        Optional[PineconeIndex]: Pinecone Index object or None if initialization fails.
    """
    global _pinecone_client, _pinecone_index, _index_name_cache

    # Use cached connection if available and matches index name
    if _pinecone_index is not None and _index_name_cache == index_name:
        logger.debug(f"Using cached Pinecone index connection: '{index_name}'")
        return _pinecone_index

    # Validate inputs
    if not api_key:
        logger.error("Pinecone API key is missing.")
        return None
    if not index_name:
         logger.error("Pinecone index name is missing.")
         return None
    if dimension <= 0:
        logger.error(f"Invalid vector dimension specified: {dimension}")
        return None

    try:
        # Initialize client only if not already initialized
        if _pinecone_client is None:
            logger.info("Initializing Pinecone connection...")
            _pinecone_client = Pinecone(api_key=api_key)
            logger.info("Pinecone client initialized.")
        else:
             logger.debug("Using existing Pinecone client connection.")

        # Check if the index exists
        logger.info(f"Checking if index '{index_name}' exists...")
        try:
             index_list: IndexList = _pinecone_client.list_indexes()
             existing_index_names = [index.name for index in index_list.indexes] # Correct way for v3+
        except AttributeError:
             logger.exception("Failed to get index names. The 'list_indexes' structure might have changed.")
             existing_index_names = [] # Assume empty if retrieval fails
        except Exception as list_err:
             logger.exception(f"An error occurred while listing indexes: {list_err}")
             return None # Cannot proceed if listing fails

        if index_name not in existing_index_names:
            logger.warning(f"Index '{index_name}' not found. Attempting to create...")
            if not spec_config or 'cloud' not in spec_config or 'region' not in spec_config:
                logger.error("Serverless spec config (cloud, region) required for creating index, but not provided.")
                return None

            try:
                logger.info(f"Creating new serverless index '{index_name}' with dimension {dimension}...")
                _pinecone_client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric='cosine', # Common choice for text embeddings
                    spec=ServerlessSpec(
                        cloud=spec_config['cloud'],
                        region=spec_config['region']
                    )
                )
                # Wait for index to be ready with timeout
                timeout_seconds = 300 # 5 minutes
                wait_interval = 10 # Check every 10 seconds
                start_wait = time.time()
                logger.info(f"Waiting up to {timeout_seconds}s for index '{index_name}' to be ready...")
                while True:
                    try:
                        index_description: IndexDescription = _pinecone_client.describe_index(index_name)
                        if index_description.status['ready']:
                            logger.info(f"Index '{index_name}' created and ready.")
                            break
                    except ApiException as desc_err:
                         # Index might not be describable immediately after create command returns
                         logger.debug(f"Describe index failed (likely temporary): {desc_err}. Retrying...")

                    if time.time() - start_wait > timeout_seconds:
                        logger.error(f"Timeout waiting for index '{index_name}' to become ready.")
                        # Consider attempting deletion of potentially stuck index here?
                        # delete_pinecone_index(api_key, index_name) # Careful with auto-delete
                        return None # Exit if index creation times out

                    time.sleep(wait_interval)

            except ApiException as create_api_err:
                 logger.exception(f"Pinecone API error creating index '{index_name}': {create_api_err}")
                 return None
            except Exception as create_err: # Catch other unexpected errors during creation
                logger.exception(f"Unexpected error creating Pinecone index '{index_name}': {create_err}")
                return None
        else:
            logger.info(f"Found existing index '{index_name}'.")

        # Connect to the index
        logger.info(f"Connecting to index '{index_name}'...")
        _pinecone_index = _pinecone_client.Index(index_name)
        _index_name_cache = index_name # Cache the name upon successful connection
        logger.info(f"Successfully connected to index '{index_name}'.")
        return _pinecone_index

    except ApiException as api_err:
         logger.exception(f"Pinecone API error during initialization/connection: {api_err}")
         _pinecone_client, _pinecone_index, _index_name_cache = None, None, None
         return None
    except Exception as e: # Catch other unexpected errors
        logger.exception(f"Unexpected error initializing Pinecone or connecting to index '{index_name}': {e}")
        # Reset cache on error
        _pinecone_client, _pinecone_index, _index_name_cache = None, None, None
        return None

# --- Upsert Function ---
def upsert_vectors(index: Optional[PineconeIndex],
                   vectors: List[Dict[str, Union[str, List[float], Dict]]], # More specific type
                   batch_size: int = 100,
                   show_progress: bool = False # Changed default to False for cleaner output
                  ) -> int:
    """
    Upserts vectors into the specified Pinecone index in batches.

    Args:
        index (Optional[PineconeIndex]): The initialized Pinecone index object.
        vectors (List[Dict]): A list of vector dictionaries for upsert, e.g.,
                               [{'id': str, 'values': List[float], 'metadata': Dict}, ...]
        batch_size (int): Number of vectors to upsert in each batch.
        show_progress (bool): Whether to log progress updates for each batch (INFO level).

    Returns:
        int: Total number of vectors successfully upserted. Returns 0 on error.
    """
    if index is None:
        logger.error("Cannot upsert: Pinecone index is not initialized.")
        return 0
    if not vectors:
        logger.warning("No vectors provided for upsert.")
        return 0

    num_vectors = len(vectors)
    upserted_count = 0
    num_batches = math.ceil(num_vectors / batch_size)

    log_level = logging.INFO if show_progress else logging.DEBUG
    logger.log(log_level, f"Starting upsert of {num_vectors} vectors in {num_batches} batches...")

    for i in range(0, num_vectors, batch_size):
        batch_start_time = time.time()
        batch = vectors[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.debug(f"Preparing batch {batch_num}/{num_batches} ({len(batch)} vectors)...")
        try:
            upsert_response = index.upsert(vectors=batch)
            # Safely get upserted count, default to 0 if key missing
            current_upserted = getattr(upsert_response, 'upserted_count', 0)
            upserted_count += current_upserted
            batch_end_time = time.time()
            logger.log(log_level, f"Batch {batch_num} upserted {current_upserted} vectors in {batch_end_time - batch_start_time:.2f}s.")

        except ApiException as api_err:
             logger.exception(f"Pinecone API error upserting batch {batch_num}: {api_err}")
             # Optionally add retry logic here or decide to stop
        except Exception as e: # Catch other unexpected errors
            logger.exception(f"Unexpected error upserting batch {batch_num}: {e}")
            # Optionally add retry logic here

        # Optional: Add a small delay between batches if hitting rate limits
        # time.sleep(0.1)

    logger.info(f"Finished upserting. Total vectors upserted: {upserted_count}/{num_vectors}")
    return upserted_count

# --- Query Function ---
def query_pinecone(index: Optional[PineconeIndex],
                   query_vector: List[float],
                   top_k: int = 3,
                   filter_dict: Optional[Dict[str, Any]] = None
                  ) -> List[Dict]:
    """
    Queries the Pinecone index to find vectors similar to the query_vector.

    Args:
        index (Optional[PineconeIndex]): The initialized Pinecone index object.
        query_vector (List[float]): The embedding vector for the query.
        top_k (int): The number of top similar results to retrieve.
        filter_dict (Optional[Dict[str, Any]]): Metadata filter to apply during query.

    Returns:
        List[Dict]: A list of matching result dictionaries (including metadata)
                    or empty list if error/no results.
    """
    if index is None:
        logger.error("Cannot query: Pinecone index is not initialized.")
        return []
    if not query_vector:
        logger.error("Cannot query: Query vector is empty.")
        return []

    logger.info(f"Querying Pinecone index '{getattr(index, 'name', 'N/A')}' with top_k={top_k}...")
    try:
        query_params = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        if filter_dict:
            query_params["filter"] = filter_dict
            logger.info(f"Applying filter: {filter_dict}")

        query_results = index.query(**query_params)
        matches = query_results.get('matches', [])
        logger.info(f"Pinecone query successful. Found {len(matches)} matches.")
        logger.debug(f"Query results (matches): {matches}") # Log detailed results at DEBUG
        return matches

    except ApiException as api_err:
        logger.exception(f"Pinecone API error during query: {api_err}")
        return []
    except Exception as e: # Catch other unexpected errors
        logger.exception(f"Unexpected error querying Pinecone index: {e}")
        return []

# --- Delete Index Function ---
def delete_pinecone_index(api_key: Optional[str], index_name: str) -> bool:
    """
    Deletes the specified Pinecone index.

    Args:
        api_key (Optional[str]): Pinecone API key (needed if client not already initialized).
        index_name (str): Name of the index to delete.

    Returns:
        bool: True if deletion was successful or index didn't exist, False otherwise.
    """
    global _pinecone_client, _pinecone_index, _index_name_cache

    logger.warning(f"Attempting to delete Pinecone index: '{index_name}'")
    if not api_key:
         logger.error("Cannot delete index: Pinecone API key is missing.")
         return False

    client_to_use = _pinecone_client
    try:
        # Initialize client if not already done (needed for list_indexes)
        if client_to_use is None:
            logger.info("Initializing temporary Pinecone client for deletion check...")
            client_to_use = Pinecone(api_key=api_key)
            logger.info("Temporary client initialized.")

        # Check if index exists before attempting deletion
        index_list = client_to_use.list_indexes()
        existing_index_names = [index.name for index in index_list.indexes]

        if index_name in existing_index_names:
            logger.info(f"Index '{index_name}' found. Proceeding with deletion...")
            client_to_use.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted successfully via API call.")
            # Clear cache if the deleted index was the cached one
            if _index_name_cache == index_name:
                 _pinecone_index = None
                 _index_name_cache = None
                 logger.debug("Cleared cached index object.")
            return True
        else:
            logger.info(f"Index '{index_name}' not found. No deletion needed.")
            return True # Return True as the desired state (no index) is achieved

    except ApiException as api_err:
        logger.exception(f"Pinecone API error deleting index '{index_name}': {api_err}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error deleting Pinecone index '{index_name}': {e}")
        return False