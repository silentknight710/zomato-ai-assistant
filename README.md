# Zomato Restaurant RAG Chatbot (Gen AI Assignment)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions about restaurants based on scraped data. It fulfills the requirements of the Zomato Generative AI Internship Assignment.

The system scrapes restaurant information (including menus, ratings, location, etc.), processes this data, stores it in a vector database (Pinecone), and uses a Large Language Model (LLM) from Hugging Face to generate answers based on retrieved context relevant to the user's query.

## Features

* Answers questions about specific restaurant details (when available in the data):
    * Cuisine types served
    * Approximate price range for two people
    * Address / Location
    * Contact number
    * Ratings and review counts
* Answers questions about specific menu items (when available in the data):
    * Price
    * Description
    * Category
* Handles basic greetings.
* Uses Pinecone for efficient vector storage and retrieval.
* Utilizes Hugging Face `transformers` and `sentence-transformers` for embedding and generation using freely available models.
* Includes configurable logging for cleaner console output during interaction.

**Current Limitations:**

* **Comparative Questions:** The chatbot struggles to reliably answer questions requiring comparison across multiple restaurants (e.g., "which restaurant is cheapest?", "compare menus"). This is due to limitations in retrieving all necessary information within the context window and the reasoning capabilities of the LLM used.
* **Retrieval Specificity:** Occasionally fails to retrieve the most relevant information for very specific menu items if they aren't semantically close enough to the query in the embedding space or aren't within the top retrieved results (`top_k`).
* **Data Dependency:** The quality and scope of answers are entirely dependent on the data scraped and present in the `data/restaurants.json` file. Missing information in the JSON will result in the chatbot being unable to answer related questions.
* **Hallucination/Accuracy:** While prompts are designed to minimize this, the LLM may occasionally misinterpret context or generate slightly inaccurate information.

## Setup Instructions

Follow these steps to set up and run the project locally.

**1. Prerequisites:**
* Python 3.8+
* `pip` (Python package installer)
* Git (for cloning the repository)
* A Pinecone account and API key (Free tier available) - [Sign up here](https://www.pinecone.io/)

**2. Clone the Repository:**
bash
git clone [https://github.com/silentknight710.git
cd [zomato-ai-assistant]

(Replace placeholders with your actual username and repository name)

**3. Create a Virtual Environment (Recommended):**

python -m venv venv
 * Activate the environment
 * On Windows (cmd):
 * venv\Scripts\activate
 * On Windows (PowerShell):
 * .\venv\Scripts\Activate.ps1
 * On macOS/Linux:
 source venv/bin/activate

**4. Install Dependencies:**

pip install -r requirements.txt

(Ensure your requirements.txt file has pinned versions based on your working environment using pip freeze)

**5. Configure Environment Variables:**

Create a file named .env in the project's root directory (alongside requirements.txt).

Add your Pinecone API key to the .env file:

* File: .env
* PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
* Optional: Specify cloud/region if different from defaults in config.py
* PINECONE_CLOUD="aws"
* PINECONE_REGION="us-east-1"
* Optional: Add Hugging Face token if needed for specific models
* HUGGINGFACE_TOKEN="YOUR_HF_TOKEN"

Important: Ensure .env is listed in your .gitignore file to avoid committing secrets.

**6. Prepare Data:**

* Run your web scraper (src/scraper.py - Note: you need to implement the scraping logic first as per the assignment) or manually place your scraped data file.

* Ensure the scraped data is saved as restaurants.json inside the data/ folder. The format should match the structure expected by src/rag_pipeline.py (the detailed format with menu items).

**7. Pinecone Index:**

The script (src/pinecone_utils.py) will attempt to create the Pinecone index specified in src/config.py (default: zomato-restaurants) if it doesn't exist, using the serverless specification defined in the config. Ensure your Pinecone account/API key has permissions to do this.

**Usage**
* Make sure your virtual environment is activated (if used) and the .env file is configured.

* Navigate to the project's root directory in your terminal.

* Run the main application script:

python src/main.py

* The script will initialize models and Pinecone. If the Pinecone index is empty, it will automatically load, process, embed, and upload the data from data/restaurants.json (this may take some time). Note: If you update data/restaurants.json, you need to manually trigger a re-population by setting FORCE_REPOPULATE = True in src/main.py for one run.

* Once initialization is complete, the chat prompt You:  will appear.

* Ask questions based on the data in your restaurants.json file. Examples:

"What cuisine does Jewel of Nizam serve?"

"What is the price range for Lake District Bar & Kitchen?"

"Where is Barbeque Spice located?"

"What is the price of the Vegetarian Biryani at Jewel of Nizam?"

"Describe the Caesar Salad at The Dining Room - Park Hyatt."

* Type quit, exit, or bye to end the session.

## Implementation Details

* **Architecture:** Standard RAG pipeline: User Query -> Embedding -> Vector DB Similarity Search (Retrieval) -> Context Augmentation -> LLM Prompting -> Answer Generation.

* **Embedding Model:** all-MiniLM-L6-v2 (from sentence-transformers) is used by default (see src/config.py). Chosen for its balance of performance and size for free-tier usage.

* **Generator Model:** google/flan-t5-base (from Hugging Face transformers) is used by default. Chosen as a capable instruction-tuned model available under Apache 2.0 license and generally usable within free compute tiers.

* **Vector Database:** Pinecone serverless index is used for storing and querying embeddings efficiently.

* **Data Processing:** Data from restaurants.json is parsed, and two types of text chunks are created for embedding (see src/rag_pipeline.py):

* General restaurant information chunks (including name, address, rating, cuisine, price range, contact).

* Individual menu item chunks (including restaurant name, category, item name, description, price, type where available).

* **Prompt Engineering:** A structured prompt with explicit instructions is used to guide the LLM (flan-t5-base) to answer based only on the retrieved context and handle cases where information is missing (see src/generator_utils.py).

* **Logging:** Python's logging module is used instead of print for status messages and errors, allowing for cleaner console output during chat interaction (controlled via src/main.py). Console output level is set to WARNING by default during chat.

## Challenges Faced

* **LLM Reasoning:** The chosen LLM (flan-t5-base) sometimes struggles with multi-step reasoning (e.g., comparisons across restaurants), complex instructions, or strictly adhering to negative constraints (e.g., not answering when context is insufficient, avoiding hallucinations) even with detailed prompting.

* **Retrieval Accuracy:** Ensuring the retrieval step fetches the most relevant chunks (e.g., general info vs. specific menu item) based on the user query semantics can be challenging. Tuning top_k and experimenting with different embedding models (all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1) yielded mixed results.

* **Context Length Limit:** Combining instructions, retrieved context (especially with top_k > 3), and the query can exceed the LLM's maximum input token limit (e.g., 512 for T5-base), requiring careful management of top_k or context chunk size.

* **Prompt Engineering:** Iteratively refining the prompt to elicit the desired behavior from the LLM while keeping it concise was a significant effort.

* **Data Scraping:** (Add challenges you faced during scraping, e.g., dynamic content, varying website structures, anti-scraping measures)

## Future Improvements

* **More Powerful LLM:** Experiment with larger models (flan-t5-large) or different architectures (e.g., Mistral/Zephyr variants) if compute resources allow, potentially improving reasoning and instruction following.

* **Code-Based Comparison:** Implement specific Python logic to handle comparison queries (e.g., "cheapest", "compare") by retrieving structured data for relevant restaurants and performing the comparison in code, rather than relying solely on the LLM.

* **Advanced Retrieval:** Explore techniques like query expansion, re-ranking retrieved results, or using metadata filters during Pinecone queries to improve retrieval relevance.

* **Conversational Memory:** Implement conversation history to allow follow-up questions.

* **Web Interface:** Build a user-friendly web interface using Streamlit or Gradio instead of the command-line interface.

* **Robust Scraping:** Enhance scraper.py with more sophisticated parsing, better error handling, and potentially rotating user agents/proxies if needed for scraping more websites reliably.

* **Metadata Filtering:** Utilize the structured metadata (e.g., price, cuisine, category) stored in Pinecone to filter search results for more targeted retrieval.
