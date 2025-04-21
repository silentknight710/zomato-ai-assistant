# src/generator_utils.py
"""
Utilities for loading the Hugging Face text generation model/pipeline
and generating the final answer based on context.
"""
import time
import logging
from typing import Optional, List, Any # For type hinting

# Attempt to import Pipeline for more specific type hinting, fallback to Any
try:
    from transformers.pipelines.base import Pipeline
except ImportError:
    Pipeline = Any # Use 'Any' if Pipeline specific import fails

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Cached Pipeline ---
# Using module-level variables for simple caching
_generator_pipeline: Optional[Pipeline] = None
_generator_model_name: Optional[str] = None

# --- Function to load the generator ---
def load_generator_pipeline(model_name: str = 'google/flan-t5-base',
                            task: str = 'text2text-generation',
                            hf_token: Optional[str] = None) -> Optional[Pipeline]:
    """
    Loads and caches the Hugging Face pipeline for text generation.

    Args:
        model_name (str): The name of the Hugging Face model.
        task (str): The task for the pipeline (e.g., 'text2text-generation').
        hf_token (Optional[str]): Hugging Face API token for private models.

    Returns:
        Optional[Pipeline]: The loaded pipeline object or None if loading fails.
    """
    global _generator_pipeline, _generator_model_name

    # Check cache first
    if _generator_pipeline is not None and _generator_model_name == model_name:
        logger.debug(f"Using cached generator pipeline: {model_name}")
        return _generator_pipeline

    logger.info(f"Loading generation pipeline: {model_name} (Task: {task})...")
    try:
        start_time = time.time()
        # Dynamically import pipeline only when needed
        from transformers import pipeline as hf_pipeline

        # Pass token only if provided
        pipeline_args = {"model": model_name, "task": task}
        if hf_token:
            logger.debug("Using Hugging Face token for pipeline.")
            pipeline_args["token"] = hf_token
            # Note: Explicit login via huggingface_hub.login(hf_token) might be needed
            # depending on model permissions and library versions. Add if required.

        _generator_pipeline = hf_pipeline(**pipeline_args)
        _generator_model_name = model_name
        end_time = time.time()
        logger.info(f"Generation pipeline '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
        return _generator_pipeline

    except ImportError:
        logger.exception("Error: 'transformers' library not found. Please install it.")
        _generator_pipeline, _generator_model_name = None, None
        return None
    except Exception as e:
        # Log the exception with traceback for detailed debugging
        logger.exception(f"Error loading generation pipeline '{model_name}': {e}")
        _generator_pipeline, _generator_model_name = None, None
        return None

# --- Function to create the prompt ---
def create_prompt(query: str, contexts: List[str]) -> str:
    """
    Creates a well-structured and clear prompt for the generator LLM
    based on the user query and retrieved contexts.
    Uses stricter instructions refined through testing.

    Args:
        query (str): The user's original question.
        contexts (List[str]): A list of context strings retrieved from the vector DB.

    Returns:
        str: The formatted prompt string.
    """
    # Handle the case where no relevant contexts were retrieved
    if not contexts:
        logger.warning(f"No relevant contexts found for query: '{query}'. Creating fallback prompt.")
        prompt = f"""You are an AI assistant. The user asked the following question:
'{query}'
However, no relevant information was found in the knowledge base to answer this question.
State that you cannot answer the question based on the information provided."""
        return prompt

    # Combine the context snippets into a single block with clear separators
    context_string = "\n\n---\n\n".join(contexts) # Use triple dashes as a clear separator

    # Construct the main prompt with clear instructions and delimiters
    prompt = f"""You are an AI assistant answering questions about restaurants based *only* on the provided context.

**Instructions:**
1. Read the 'Provided Context' below carefully.
2. Answer the 'User Question' based *solely* on the information found within the 'Provided Context'.
3. Do not use any outside knowledge or make assumptions.
4. If the 'Provided Context' does not contain the information needed to answer the 'User Question', state exactly: 'Based on the provided information, I cannot answer that question.'
5. Be concise and directly answer the question using only information from the context.
6. Do NOT include context separators like '---' or repeat parts of the user question or these instructions in your answer.

**Provided Context:**
--- Start of Context ---
{context_string}
--- End of Context ---

**User Question:** {query}

**Answer:** """

    # Log the generated prompt at DEBUG level (won't show unless configured)
    logger.debug(f"\n--- Generated Prompt ---\n{prompt}\n--- End of Prompt ---")

    return prompt

# --- Function to generate the answer ---
def generate_answer(generator: Optional[Pipeline],
                    prompt: str,
                    max_length: int = 150,
                    min_length: int = 5, # Slightly increased min_length
                    repetition_penalty: float = 1.2
                   ) -> str:
    """
    Generates an answer using the loaded generator pipeline and the prompt.

    Args:
        generator (Optional[Pipeline]): The loaded generator pipeline object.
        prompt (str): The formatted prompt including context and query.
        max_length (int): Maximum length of the generated answer tokens.
        min_length (int): Minimum length of the generated answer tokens.
        repetition_penalty (float): Penalty for repeating tokens ( > 1.0 discourages).

    Returns:
        str: The generated answer string, or an error message.
    """
    if generator is None:
        logger.error("Generation pipeline is not loaded. Cannot generate answer.")
        return "Sorry, I cannot generate an answer right now due to a configuration issue."

    logger.info("Generating answer...")
    try:
        start_time = time.time()

        # Generate the answer using specified parameters
        generated_output = generator(
            prompt,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=1,
            do_sample=False, # Keep deterministic output unless sampling needed
            repetition_penalty=repetition_penalty
            # Consider adding other parameters if needed (e.g., temperature, top_p if do_sample=True)
        )

        # Extract and clean the answer text
        answer = generated_output[0]['generated_text'].strip()
        # Basic post-processing: remove potential prompt artifacts
        if answer.startswith("Answer:"):
            answer = answer.replace("Answer:", "").strip()

        end_time = time.time()
        logger.info(f"Answer generated in {end_time - start_time:.2f} seconds.")
        logger.debug(f"Generated Answer: {answer}") # Log the answer at DEBUG level
        return answer

    except Exception as e:
        logger.exception(f"Error during answer generation: {e}")
        return "Sorry, I encountered an error while trying to generate the answer."