# src/rag_pipeline.py
"""
Orchestrates the RAG (Retrieval-Augmented Generation) pipeline.

Handles loading and processing restaurant data from JSON, embedding and
storing it in Pinecone, and executing the query-retrieve-generate sequence.
"""

import json
import time
import os
import logging
from typing import List, Dict, Any, Optional, TypedDict, Union

# Define a type alias for the chunk structure for clarity
class ChunkMetadata(TypedDict, total=False): # Allow partial keys
    type: str
    source_text: str
    restaurant_name: str
    address: str
    rating_value: Optional[Union[str, float, int]]
    rating_count: Optional[int]
    cuisine: Optional[str]
    price_range: Optional[str]
    telephone: Optional[str]
    category: Optional[str]
    item_name: Optional[str]
    description: Optional[str]
    price: Optional[str] # Formatted price string
    item_type: Optional[str] # e.g., Vegetarian

class Chunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata

# Configure logging for this module
logger = logging.getLogger(__name__)

# Import utility functions from other modules within the src directory
# Assuming these modules also use logging appropriately now/later
try:
    from .embedding_utils import load_embedding_model, generate_embeddings
    from .pinecone_utils import init_pinecone, upsert_vectors, query_pinecone
    from .generator_utils import load_generator_pipeline, create_prompt, generate_answer
    # Assuming embedding_model and generator_pipeline types are complex, use Any for now
    # Or import specific types if available and stable
    EmbeddingModel = Any
    GeneratorPipeline = Any
    PineconeIndex = Any # Replace with actual type if pinecone library provides it
except ImportError: # Fallback for running script directly
    logger.warning("Running rag_pipeline.py directly or imports failed, using local imports.")
    from embedding_utils import load_embedding_model, generate_embeddings
    from pinecone_utils import init_pinecone, upsert_vectors, query_pinecone
    from generator_utils import load_generator_pipeline, create_prompt, generate_answer
    EmbeddingModel = Any
    GeneratorPipeline = Any
    PineconeIndex = Any

# --- Helper function for parsing general info ---
def _parse_general_info(restaurant_data: Dict[str, Any], restaurant_id: str) -> Optional[Chunk]:
    """Parses general restaurant info and creates a chunk."""
    restaurant_name = restaurant_data.get("name")
    if not restaurant_name:
        logger.warning(f"Skipping entry with missing name (URL: {restaurant_data.get('url')}).")
        return None

    address_obj = restaurant_data.get("address", {})
    address_parts = [address_obj.get("street"), address_obj.get("locality"), address_obj.get("region"), address_obj.get("postalCode"), address_obj.get("country")]
    address_full = ", ".join(filter(None, address_parts))

    rating_obj = restaurant_data.get("rating", {})
    rating_value = rating_obj.get("value")
    rating_count = rating_obj.get("count")
    rating_text = f"{rating_value}/5.0 ({rating_count} reviews)" if rating_value and rating_count else (str(rating_value) if rating_value else None)

    cuisine = restaurant_data.get("cuisine")
    price_range = restaurant_data.get("price_range")
    telephone = restaurant_data.get("telephone")

    general_chunk_text = f"General information for Restaurant: {restaurant_name}."
    details_list = []
    if address_full: details_list.append(f"Address is {address_full}")
    if rating_text: details_list.append(f"Overall rating is {rating_text}")
    if cuisine: details_list.append(f"Serves cuisines like {cuisine}")
    if price_range: details_list.append(f"Typical price range is {price_range}")
    if telephone: details_list.append(f"Contact number is {telephone}")
    if details_list:
        general_chunk_text += " Details: " + "; ".join(details_list) + "."

    metadata: ChunkMetadata = {"type": "general_info", "source_text": general_chunk_text}
    metadata["restaurant_name"] = restaurant_name
    if address_full: metadata["address"] = address_full
    if rating_value is not None: metadata["rating_value"] = rating_value
    if rating_count is not None: metadata["rating_count"] = rating_count
    if cuisine is not None: metadata["cuisine"] = cuisine
    if price_range is not None: metadata["price_range"] = price_range
    if telephone is not None: metadata["telephone"] = telephone

    return {'id': f"{restaurant_id}_info", 'text': general_chunk_text, 'metadata': metadata}

# --- Helper function for parsing menu items ---
def _parse_menu_items(restaurant_data: Dict[str, Any], restaurant_id: str, restaurant_name: str) -> List[Chunk]:
    """Parses menu items from restaurant data and creates chunks."""
    menu_chunks = []
    menu_item_counter = 0
    menu_list = restaurant_data.get("menu", [])

    if not (menu_list and isinstance(menu_list, list) and len(menu_list) > 0):
        logger.debug(f"No menu list found for restaurant: {restaurant_name}")
        return []

    # Assuming the actual menu is within the first element's 'menu' key
    menu_data = menu_list[0].get("menu", {})
    if not isinstance(menu_data, dict):
        logger.warning(f"Unexpected menu data format for restaurant: {restaurant_name}")
        return []

    for category, items in menu_data.items():
        if not isinstance(items, list):
            logger.debug(f"Skipping non-list items under category '{category}' for {restaurant_name}")
            continue

        for item in items:
            item_name, item_desc, item_price_str, item_type = None, None, "N/A", None

            if isinstance(item, dict):
                item_name = item.get("name")
                item_desc = item.get("description")
                price_val = item.get("price", item.get("price_inr"))
                if isinstance(price_val, list): item_price_str = f"₹{price_val[0]} - ₹{price_val[1]}"
                elif price_val is not None: item_price_str = f"₹{price_val}"
                item_type = item.get("type")
            elif isinstance(item, str):
                item_name = item

            if not item_name: continue # Skip item if it has no name

            menu_chunk_text = f"Menu item at {restaurant_name}: {item_name}."
            details_list = [f"Category: {category}"]
            if item_desc: details_list.append(f"Description: {item_desc}")
            if item_price_str != "N/A": details_list.append(f"Price: {item_price_str}")
            if item_type: details_list.append(f"Type: {item_type}")
            menu_chunk_text += " Details: " + "; ".join(details_list) + "."

            menu_metadata: ChunkMetadata = {"type": "menu_item", "source_text": menu_chunk_text}
            menu_metadata["restaurant_name"] = restaurant_name
            if category: menu_metadata["category"] = category
            menu_metadata["item_name"] = item_name
            if item_desc: menu_metadata["description"] = item_desc
            if item_price_str != "N/A": menu_metadata["price"] = item_price_str
            if item_type: menu_metadata["item_type"] = item_type

            menu_chunks.append({
                'id': f"{restaurant_id}_menu_{menu_item_counter}",
                'text': menu_chunk_text,
                'metadata': menu_metadata
            })
            menu_item_counter += 1

    return menu_chunks

# --- Main Data Loading and Chunking Function ---
def load_and_chunk_data(file_path: str) -> List[Chunk]:
    """
    Loads detailed restaurant data from JSON, processes it using helper
    functions, and returns a list of structured chunks.

    Args:
        file_path (str): Path to the JSON data file.

    Returns:
        List[Chunk]: A list of dictionaries, where each dict represents a
                     data chunk with 'id', 'text', and 'metadata'.
                     Returns empty list on error.
    """
    logger.info(f"Attempting to load and chunk data from: {file_path}")
    script_dir = os.path.dirname(__file__)
    absolute_file_path = os.path.abspath(os.path.join(script_dir, file_path))
    logger.debug(f"Absolute path calculated: {absolute_file_path}")

    if not os.path.exists(absolute_file_path):
        logger.error(f"Data file not found at '{absolute_file_path}'.")
        logger.error("Ensure the JSON file exists in the 'data' folder relative to 'src'.")
        return []

    try:
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            restaurants_data = json.load(f)
        logger.info(f"Successfully loaded data for {len(restaurants_data)} restaurant entries.")
    except FileNotFoundError: # More specific error
        logger.exception(f"Data file not found error for '{absolute_file_path}'.")
        return []
    except json.JSONDecodeError:
        logger.exception(f"Could not decode JSON from '{absolute_file_path}'. Check file format.")
        return []
    except Exception as e: # Catch other potential loading errors
        logger.exception(f"An unexpected error occurred loading data file '{absolute_file_path}': {e}")
        return []

    all_data_chunks: List[Chunk] = []
    global_chunk_counter = 0 # Use a global counter if needed for unique IDs across files

    for i, restaurant_data in enumerate(restaurants_data):
        restaurant_id = restaurant_data.get("id", f"res_{i}") # Use provided ID or index

        # Create general info chunk
        general_chunk = _parse_general_info(restaurant_data, restaurant_id)
        if general_chunk:
            all_data_chunks.append(general_chunk)
            global_chunk_counter += 1

        # Create menu item chunks
        restaurant_name = restaurant_data.get("name", "Unknown Restaurant") # Needed for menu items
        menu_chunks = _parse_menu_items(restaurant_data, restaurant_id, restaurant_name)
        all_data_chunks.extend(menu_chunks)
        global_chunk_counter += len(menu_chunks)

    logger.info(f"Created {len(all_data_chunks)} total data chunks.")
    if len(all_data_chunks) == 0 and len(restaurants_data) > 0:
         logger.warning("No valid chunks created from JSON data. Check structure and required fields.")
    return all_data_chunks

# --- Function to Setup/Populate Pinecone ---
def setup_and_populate_pinecone(config: Any, # Replace Any with actual config type if defined
                                embedding_model: Optional[EmbeddingModel],
                                force_repopulate: bool = False) -> Optional[PineconeIndex]:
    """
    Initializes Pinecone index and populates it with data if it's empty
    or if force_repopulate is True.

    Args:
        config: Configuration object/module with Pinecone/data settings.
        embedding_model: The loaded sentence transformer model.
        force_repopulate (bool): If True, clears existing index and repopulates.

    Returns:
        Optional[PineconeIndex]: Pinecone Index object or None if setup fails.
    """
    if embedding_model is None:
        logger.error("Cannot setup Pinecone: Embedding model not loaded.")
        return None

    try:
        # Assuming get_sentence_embedding_dimension is a method on the model object
        embedding_dim: int = embedding_model.get_sentence_embedding_dimension()
        if embedding_dim <= 0:
             logger.error("Invalid embedding dimension obtained from model.")
             return None
    except AttributeError:
         logger.exception("Failed to get embedding dimension from the provided model object.")
         return None

    # Initialize Pinecone connection
    pinecone_index = init_pinecone(
        api_key=config.PINECONE_API_KEY,
        index_name=config.PINECONE_INDEX_NAME,
        dimension=embedding_dim,
        spec_config=config.PINECONE_SERVERLESS_SPEC
    )

    if pinecone_index is None:
        logger.error("Failed to initialize Pinecone index connection.")
        return None

    try:
        # Check index status and decide whether to populate
        index_stats = pinecone_index.describe_index_stats()
        # Access vector_count safely using get, default to 0
        vector_count = index_stats.get('total_vector_count', 0)
        logger.info(f"Pinecone index '{config.PINECONE_INDEX_NAME}' current vector count: {vector_count}")

        if force_repopulate:
            logger.warning(f"FORCE_REPOPULATE=True. Clearing all vectors from index '{config.PINECONE_INDEX_NAME}'...")
            try:
                pinecone_index.delete(delete_all=True)
                logger.info("Index cleared successfully. Waiting briefly...")
                time.sleep(5)
                vector_count = 0 # Reset count after successful clear
            except Exception as delete_err:
                 logger.exception(f"Error clearing Pinecone index: {delete_err}")
                 # Decide whether to proceed or exit if clearing fails
                 return None # Exit if clearing fails

        if vector_count == 0:
            logger.info("Pinecone index is empty or cleared. Populating with data...")

            # 1. Load and Chunk Data (using the refactored function)
            data_chunks = load_and_chunk_data(config.RESTAURANT_DATA_PATH)
            if not data_chunks:
                logger.error("No data chunks loaded or created from JSON. Cannot populate index.")
                return None # Return None if loading/chunking fails

            # 2. Generate Embeddings
            texts_to_embed = [chunk['text'] for chunk in data_chunks]
            embeddings = generate_embeddings(embedding_model, texts_to_embed) # Assumes this returns List[List[float]]
            if not embeddings or len(embeddings) != len(data_chunks):
                logger.error("Failed to generate embeddings or count mismatch. Cannot populate index.")
                return None # Return None if embedding fails

            # 3. Prepare Vectors for Pinecone (using the Chunk TypedDict structure)
            vectors_to_upsert = [
                {'id': chunk['id'], 'values': embedding, 'metadata': chunk['metadata']}
                for chunk, embedding in zip(data_chunks, embeddings)
            ]

            # 4. Upsert to Pinecone
            logger.info(f"Attempting to upsert {len(vectors_to_upsert)} vectors...")
            upserted_count = upsert_vectors(pinecone_index, vectors_to_upsert) # Assumes this returns int
            if upserted_count > 0:
                 logger.info(f"Successfully populated index with {upserted_count} vectors.")
                 logger.info("Waiting a few seconds for index to update...")
                 time.sleep(10) # Give index time to become consistent
            else:
                 logger.error("Failed to upsert vectors into Pinecone. Index might be partially populated or empty.")
                 # Return None as population step failed
                 return None

        else:
            logger.info("Pinecone index already contains data. Skipping population.")

        return pinecone_index

    except Exception as e: # Catch errors during stats check or population logic
        logger.exception(f"An error occurred during Pinecone setup/population phase: {e}")
        return None

# --- Function to Get Chatbot Response ---
def get_chatbot_response(query: str,
                         embedding_model: Optional[EmbeddingModel],
                         pinecone_index: Optional[PineconeIndex],
                         generator_pipeline: Optional[GeneratorPipeline],
                         top_k: int = 3) -> str:
    """
    Handles the end-to-end RAG process for a single user query.
    Embeds query, retrieves context, creates prompt, generates answer.

    Args:
        query (str): The user's question.
        embedding_model: Loaded sentence transformer model.
        pinecone_index: Initialized Pinecone index object.
        generator_pipeline: Loaded Hugging Face generation pipeline.
        top_k (int): Number of results to retrieve from Pinecone.

    Returns:
        str: The chatbot's generated response or an error message.
    """
    start_time = time.time()
    logger.info(f"Processing query: '{query}'")

    # --- Validate Inputs ---
    if not all([query, embedding_model, pinecone_index, generator_pipeline]):
        logger.error("Cannot get response: Missing one or more required components (query, models, index).")
        return "Sorry, the chatbot is not properly configured."

    # --- 1. Embed the Query ---
    query_embedding_list = generate_embeddings(embedding_model, [query]) # Expects list, returns list
    if not query_embedding_list:
        logger.error("Could not generate embedding for the query.")
        return "Sorry, I couldn't understand your query."
    query_embedding: List[float] = query_embedding_list[0] # Get the single embedding vector

    # --- 2. Retrieve Relevant Context ---
    logger.info("Retrieving relevant context from Pinecone...")
    retrieved_matches = query_pinecone(pinecone_index, query_embedding, top_k=top_k) # Assumes returns list of matches

    contexts: List[str] = []
    if not retrieved_matches:
        logger.warning("No relevant context found in Pinecone for this query.")
    else:
        # Extract the 'source_text' from metadata for the generator prompt
        contexts = [match['metadata'].get('source_text', '') # Default to empty string if missing
                    for match in retrieved_matches]
        contexts = [ctx for ctx in contexts if ctx] # Filter out potential empty strings
        logger.info(f"Retrieved {len(contexts)} non-empty context(s).")

        # Log retrieved context at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- Retrieved Contexts ---")
            for i, ctx in enumerate(contexts):
                logger.debug(f"Context {i+1}:\n{ctx}\n---")

    # --- 3. Create Prompt ---
    # Ensure using the refined create_prompt function from generator_utils
    prompt = create_prompt(query, contexts)

    # --- 4. Generate Answer ---
    final_answer = generate_answer(generator_pipeline, prompt) # Use refined generate_answer

    end_time = time.time()
    logger.info(f"Total processing time for query: {end_time - start_time:.2f} seconds.")
    return final_answer