# src/config.py
"""
Configuration settings for the RAG Chatbot.

Reads sensitive information like API keys from environment variables
and defines constants for models, data paths, and RAG parameters.
Uses python-dotenv to load environment variables from a .env file if present.
"""

import os
import logging
from typing import Dict, Final, Optional, Union # For type hinting

from dotenv import load_dotenv

# Configure basic logging FOR THIS MODULE ONLY (optional, helpful for debugging config loading)
# A more comprehensive logging setup will be in main.py
# logging.basicConfig(level=logging.INFO) # Uncomment for debugging config loading itself
# logger = logging.getLogger(__name__) # Get a logger specific to this module

# Load variables from a .env file if it exists
# Searches in the current directory or parent directories.
load_dotenv()
# logger.info(".env file loaded if found.") # Example debug log

# --- Pinecone Configuration ---
# Read the API key from an environment variable named 'PINECONE_API_KEY'
PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME: Final[str] = "zomato-restaurants" # Choose your index name

# Specification for creating a serverless index (needed only for index creation)
# Type hinting Dict[str, str] indicates a dictionary where keys and values are strings
PINECONE_SERVERLESS_SPEC: Dict[str, str] = {
    "cloud": os.getenv("PINECONE_CLOUD", "aws"),
    "region": os.getenv("PINECONE_REGION", "us-east-1")
}

# --- Model Configuration ---
# Using Final[str] indicates these are intended as constants
EMBEDDING_MODEL_NAME: Final[str] = 'all-MiniLM-L6-v2'
GENERATOR_MODEL_NAME: Final[str] = 'google/flan-t5-base'

# --- RAG Configuration ---
TOP_K_RESULTS: Final[int] = 3

# --- Hugging Face Configuration (Optional) ---
HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN", None)

# --- Data Configuration ---
RESTAURANT_DATA_PATH: Final[str] = "../data/restaurants.json"

# --- Validation (Using Logging) ---
# Note: These log messages will only appear if the root logger is configured
# in main.py to handle WARNING/ERROR levels appropriately.
if not PINECONE_API_KEY:
    logging.error("❌ FATAL ERROR: Pinecone API Key is not set.")
    logging.error("   Please set the PINECONE_API_KEY environment variable (e.g., in a .env file).")
    # Consider adding sys.exit(1) here if the key is absolutely essential to even start
    # import sys
    # sys.exit(1)
elif len(PINECONE_API_KEY) < 20: # Basic sanity check
    logging.warning(f"⚠️ WARNING: PINECONE_API_KEY seems very short ('{PINECONE_API_KEY[:5]}...'). Is it correct?")
else:
    logging.info("Pinecone API Key found.") # Log success at INFO level

# Example of logging other loaded configs at INFO level
# logging.info(f"Pinecone Index Name: {PINECONE_INDEX_NAME}")
# logging.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
# logging.info(f"Generator Model: {GENERATOR_MODEL_NAME}")