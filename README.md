# PDF RAG System

Answer any question about a PDF document using AI.

## How it works
1. Loads PDF and splits into chunks
2. Converts chunks to vector embeddings (HuggingFace)
3. Stores in ChromaDB for similarity search
4. User question → finds relevant chunks → Groq LLM answers with source pages

## Tech Stack
- Python, LangChain, ChromaDB, HuggingFace Embeddings, Groq API

## Setup
1. Clone the repo
2. Run: `pip install -r requirements.txt`
3. Create `.env` file with: `GROQ_API_KEY=your_key_here`
4. Update `FILE_PATH` in `rag1.py` to your PDF path
5. Run: `python rag1.py`