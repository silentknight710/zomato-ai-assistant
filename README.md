# Zomato Restaurant RAG Chatbot (Gen AI Assignment)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed to answer user questions about restaurants based on scraped data. It fulfills the requirements of the Zomato Generative AI Internship Assignment.

The system scrapes restaurant information (including menus, ratings, location, etc.), processes this data, stores it in a vector database (**Pinecone**), and uses a **Large Language Model (LLM)** from **Hugging Face** (`google/flan-t5-base`) to generate answers based on retrieved context relevant to the user's query.

---

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
* Handles basic greetings directly without involving the RAG pipeline.
* Uses **Pinecone** (serverless index) for efficient vector storage and retrieval.
* Utilizes Hugging Face `transformers` and `sentence-transformers` for embedding and generation using freely available models.
* Includes configurable **logging** for cleaner console output during interaction (defaults to showing only WARNING or higher on console).

---

## Current Limitations

* **Comparative Questions:** The chatbot struggles to reliably answer questions requiring comparison across multiple restaurants (e.g., "which restaurant is cheapest?", "compare menus"). This is due to limitations in retrieving all necessary information within the context window and the reasoning capabilities of the LLM used.
* **Retrieval Specificity:** Occasionally fails to retrieve the most relevant information for very specific menu items if they aren't semantically close enough to the query in the embedding space or aren't within the top retrieved results (`top_k=3`).
* **Data Dependency:** The quality and scope of answers are entirely dependent on the data scraped and present in the `data/restaurants.json` file. Missing information in the JSON will result in the chatbot being unable to answer related questions.
* **Hallucination/Accuracy:** While prompts are designed to minimize this, the LLM (`flan-t5-base`) may occasionally misinterpret context or generate slightly inaccurate information, especially for complex queries or when retrieval provides suboptimal context.

---

## Project Structure


zomato_chatbot/
├── data/
│   └── restaurants.json       # Scraped restaurant data (MUST BE PROVIDED)
├── src/                     # Source code
│   ├── init.py
│   ├── config.py            # Configuration (API keys, models, paths)
│   ├── embedding_utils.py   # Embedding model loading & generation
│   ├── generator_utils.py   # LLM loading, prompt creation, answer generation
│   ├── main.py              # Main application entry point, chat loop
│   ├── pinecone_utils.py    # Pinecone connection, upsert, query, etc.
│   └── rag_pipeline.py      # RAG orchestration, data loading/chunking
├── .env                     # Store API keys here (add to .gitignore!)
├── .gitignore               # Specifies intentionally untracked files
├── requirements.txt         # Python dependencies
└── README.md                # This file


---

## Setup Instructions

Follow these steps to set up and run the project locally.

**1. Prerequisites:**
* Python 3.8+
* `pip` *(Python package installer)*
* Git *(for cloning the repository)*
* A Pinecone account and API key *(Free tier available)* - [Sign up here](https://www.pinecone.io/)

**2. Clone the Repository:**
bash
git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

(Replace placeholders [Your-GitHub-Username] and [Your-Repo-Name] with your actual details)

3. Create a Virtual Environment (Recommended):

python -m venv venv
# Activate the environment
# On Windows (cmd):
# venv\Scripts\activate
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate

4. Install Dependencies:

pip install -r requirements.txt

(Ensure your requirements.txt file has pinned versions based on your working environment using pip freeze)

5. Configure Environment Variables:

Create a file named .env in the project's root directory (the same directory as requirements.txt).

Add your Pinecone API key to the .env file:

# File: .env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
# Optional: Specify cloud/region if different from defaults in config.py
# PINECONE_CLOUD="aws"
# PINECONE_REGION="us-east-1"
# Optional: Add Hugging Face token if needed for specific models
# HUGGINGFACE_TOKEN="YOUR_HF_TOKEN"

Important: Ensure .env is listed in your .gitignore file to avoid committing secrets.

6. Prepare Data:

Run your web scraper (src/scraper.py - Note: you need to implement the scraping logic first as per the assignment) or manually place your scraped data file.

Ensure the scraped data is saved as restaurants.json inside the data/ folder. The format should match the detailed structure expected by src/rag_pipeline.py.

7. Pinecone Index:

The script (src/pinecone_utils.py) will attempt to create the Pinecone index specified in src/config.py (default: zomato-restaurants) if it doesn't exist. Ensure your Pinecone account/API key has permissions for index creation.

Usage
1. Activate Environment & Configure:

Make sure your virtual environment is activated (if used) and the .env file is configured with your Pinecone API key.

2. Navigate to Project Directory:

Open your terminal or command prompt and navigate to the project's root directory (zomato_chatbot/).

3. Run the Application:

Execute the main script:

python src/main.py

4. Initialization & Data Loading:

The script will initialize models and Pinecone.

If the Pinecone index is empty (or if FORCE_REPOPULATE is set to True in src/main.py), it will automatically load data from data/restaurants.json, process it, generate embeddings, and upload them to Pinecone. (This may take some time, especially the first time or with large data).

Note: If you update data/restaurants.json after the initial population, you must manually trigger a re-population by setting FORCE_REPOPULATE = True in src/main.py for one run, then set it back to False.

5. Interact with Chatbot:

Once initialization is complete (you'll see "Initialization complete..." logged if INFO level is enabled, otherwise it will go straight to the prompt), the chat prompt You:  will appear.

Ask questions based on the data in your restaurants.json file. Examples:

"What cuisine does Jewel of Nizam serve?"

"What is the price range for Lake District Bar & Kitchen?"

"Where is Barbeque Spice located?"

"What is the price of the Vegetarian Biryani at Jewel of Nizam?"

"Describe the Caesar Salad at The Dining Room - Park Hyatt."

6. Exit:

Type quit, exit, or bye to end the session.

Implementation Details
Architecture: Standard RAG pipeline: User Query -> Embedding -> Vector DB Similarity Search (Retrieval) -> Context Augmentation -> LLM Prompting -> Answer Generation.

Embedding Model: all-MiniLM-L6-v2 (from sentence-transformers) is used by default (see src/config.py).

Generator Model: google/flan-t5-base (from Hugging Face transformers) is used by default.

Vector Database: Pinecone serverless index.

Data Processing: Data from restaurants.json is parsed into two chunk types: general restaurant info and individual menu items (see src/rag_pipeline.py). Metadata includes structured fields like name, category, price, etc.

Prompt Engineering: A structured prompt with explicit instructions guides the LLM to answer based only on retrieved context (see src/generator_utils.py).

Logging: Python's logging module handles status/error messages. Console level set in src/main.py defaults to WARNING to keep chat interaction clean.

Challenges Faced
LLM Reasoning: The flan-t5-base model sometimes struggles with multi-step reasoning (e.g., comparisons), complex instructions, or strictly adhering to negative constraints (avoiding hallucination) even with detailed prompting.

Retrieval Accuracy: Ensuring retrieval of the most relevant chunk type (general info vs. menu item) based on query semantics was challenging. Tuning top_k and embedding models yielded varying results. Context length limits also constrain retrieval.

Prompt Engineering: Iteratively refining the prompt for clarity, context grounding, and instruction following required significant experimentation.

Data Scraping: [Add 1-2 sentences about specific challenges you faced during scraping, e.g., handling websites with JavaScript, inconsistent HTML structures, avoiding blocks, extracting specific fields like price/features reliably.]

Future Improvements
More Powerful LLM: Experiment with larger models (flan-t5-large) or different architectures (Mistral, Zephyr) if compute resources allow.

Code-Based Comparison: Implement Python logic to handle comparison queries robustly outside the LLM.

Advanced Retrieval: Explore query expansion, re-ranking, or metadata filtering to improve retrieval relevance.

Conversational Memory: Add history to handle follow-up questions.

Web Interface: Build a Streamlit or Gradio UI.

Robust Scraping: Enhance scraper.py further.

Metadata Filtering: Implement query logic that uses Pinecone's metadata filtering capabilities.

Demo Video
*[Link to your 3-minute demo video
