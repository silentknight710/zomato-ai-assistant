# src/main.py
"""
Main entry point for the Zomato RAG Chatbot.

Initializes logging, configurations, AI models, vector database connection,
and starts the interactive command-line chat loop.
"""

import time
import sys
import os
import logging
import logging.config
from typing import List, Optional, Any

# --- Type Hinting Imports (use Any for complex external library types) ---
# Attempt to import specific types if possible for better hinting
try:
    from sentence_transformers import SentenceTransformer
    EmbeddingModel = SentenceTransformer
except ImportError:
    EmbeddingModel = Any
try:
    from transformers.pipelines.base import Pipeline
    GeneratorPipeline = Pipeline
except ImportError:
    GeneratorPipeline = Any
try:
    # Assuming pinecone library exposes an Index type, otherwise use Any
    from pinecone.index import Index as PineconeIndexType # Example, adjust if needed
    PineconeIndex = PineconeIndexType
except ImportError:
    PineconeIndex = Any

# --- Logging Configuration ---
def setup_logging(default_level=logging.INFO, console_level=logging.WARNING):
    """Configures logging for the application."""
    # Basic configuration (can be expanded to use dictConfig for file logging etc.)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=default_level, format=log_format)

    # Configure console handler level separately if needed
    # Find the console handler and set its level
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler): # Usually console handler
            # Set console level higher to suppress INFO/DEBUG during chat
            handler.setLevel(console_level)
            handler.setFormatter(logging.Formatter(log_format)) # Ensure format is applied
            logging.info(f"Console log level set to: {logging.getLevelName(console_level)}")
            break
    else: # If no stream handler found by default, add one
         console_handler = logging.StreamHandler(sys.stdout)
         console_handler.setLevel(console_level)
         console_handler.setFormatter(logging.Formatter(log_format))
         root_logger.addHandler(console_handler)
         logging.info(f"Added console handler with level: {logging.getLevelName(console_level)}")

# Get a logger for this main script
logger = logging.getLogger(__name__)

# --- Adjust path and Import Modules ---
# Assuming src is in the same directory or PYTHONPATH includes the parent
try:
    import config # Your configuration file
    from embedding_utils import load_embedding_model
    from pinecone_utils import init_pinecone
    from generator_utils import load_generator_pipeline
    from rag_pipeline import setup_and_populate_pinecone, get_chatbot_response
except ImportError as e:
    # Log critical error if imports fail
    logging.critical(f"Error importing necessary modules. Ensure files are in 'src' directory.")
    logging.critical(f"Check dependencies in requirements.txt. Details: {e}")
    sys.exit(1) # Exit if core components can't be imported

# --- Constants ---
GREETINGS: List[str] = ["hi", "hello", "hey", "greetings", "yo"]

# --- Chat Loop Function ---
def run_chat_loop(embedding_model: EmbeddingModel,
                  pinecone_index: PineconeIndex,
                  generator_pipeline: GeneratorPipeline):
    """
    Runs the interactive command-line chat loop.

    Args:
        embedding_model: The loaded embedding model object.
        pinecone_index: The initialized Pinecone index object.
        generator_pipeline: The loaded text generation pipeline object.
    """
    # Use standard print for direct user interaction prompts/messages
    print("\n--- Zomato Restaurant Chatbot ---")
    print("Ask questions about restaurants in the knowledge base.")
    print("Type 'quit', 'exit', or 'bye' to exit.")

    while True:
        try:
            user_input: str = input("\nYou: ") # Get user input
            user_input_lower: str = user_input.lower().strip()

            # Handle exit commands
            if user_input_lower in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye!") # Use print for final user message
                break

            # Handle empty input
            if not user_input_lower:
                continue

            # Handle simple greetings directly
            if user_input_lower in GREETINGS:
                print("Chatbot: Hello! How can I help you find restaurant information?") # Use print for canned response
                continue # Skip RAG pipeline for greetings

            # --- Process query using RAG pipeline ---
            # Pass original case query to RAG pipeline
            response: str = get_chatbot_response(
                query=user_input,
                embedding_model=embedding_model,
                pinecone_index=pinecone_index,
                generator_pipeline=generator_pipeline,
                top_k=config.TOP_K_RESULTS # Get top_k from config
            )

            # --- Print final chatbot response ---
            print(f"Chatbot: {response}") # Use print for the final answer display

        except (EOFError, KeyboardInterrupt): # Handle Ctrl+D or Ctrl+C
            print("\nChatbot: Exiting...") # Use print for exit message
            break
        except Exception as e:
            # Log unexpected errors in the loop with traceback
            logger.exception(f"An unexpected error occurred in the chat loop: {e}")
            # Inform the user without revealing detailed error
            print("Chatbot: Sorry, an unexpected error occurred. Please try again.")

# --- Main Execution Function ---
def main():
    """Initializes all components and runs the chatbot application."""
    # Setup logging first - CONSOLE_LEVEL controls verbosity during chat
    # Set default_level=logging.DEBUG to see detailed logs from all modules
    # Set console_level=logging.WARNING to only see warnings/errors on console
    setup_logging(default_level=logging.INFO, console_level=logging.WARNING)

    logger.info("===================================")
    logger.info("Initializing Chatbot Components...")
    start_time = time.time()

    # --- 1. Load Embedding Model ---
    logger.info("--- Step 1: Loading Embedding Model ---")
    embedding_model, embedding_dim = load_embedding_model(config.EMBEDDING_MODEL_NAME)
    if embedding_model is None or embedding_dim <= 0:
        logger.critical("Fatal Error: Could not load embedding model or get valid dimension. Exiting.")
        sys.exit(1) # Exit if embedding model fails

    # --- 2. Initialize Pinecone & Populate (if needed) ---
    logger.info("--- Step 2: Initializing Pinecone and Knowledge Base ---")
    # Set force_repopulate=True only when you need to clear and reload data
    FORCE_REPOPULATE: bool = False
    pinecone_index = setup_and_populate_pinecone(
        config=config,
        embedding_model=embedding_model,
        force_repopulate=FORCE_REPOPULATE
    )
    if pinecone_index is None:
        logger.critical("Fatal Error: Could not initialize or populate Pinecone index. Exiting.")
        sys.exit(1) # Exit if index setup fails

    # --- 3. Load Generator Pipeline ---
    logger.info("--- Step 3: Loading Text Generation Pipeline ---")
    generator_pipeline = load_generator_pipeline(
        model_name=config.GENERATOR_MODEL_NAME,
        hf_token=config.HUGGINGFACE_TOKEN # Pass token if available in config
    )
    if generator_pipeline is None:
        logger.critical("Fatal Error: Could not load generation pipeline. Exiting.")
        sys.exit(1) # Exit if generator fails

    # --- Initialization Complete ---
    end_time = time.time()
    logger.info(f"Initialization complete in {end_time - start_time:.2f} seconds.")
    logger.info("-----------------------------------")

    # --- 4. Start Chat Loop ---
    try:
         run_chat_loop(embedding_model, pinecone_index, generator_pipeline)
    except Exception as loop_err: # Catch unexpected errors starting the loop
         logger.exception(f"Error starting or running the chat loop: {loop_err}")
    finally:
         # You could add cleanup code here if needed (e.g., closing connections)
         logger.info("Chatbot session ended.")
         print("\nChatbot session ended.") # Also print for user visibility

# --- Script Entry Point ---
if __name__ == "__main__":
    main()