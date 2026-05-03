from dotenv import load_dotenv
load_dotenv()
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# --- 1. Configuration ---
# --- 1. Configuration ---


FILE_PATH = r"C:\Users\shiva\OneDrive\Desktop\print\MP\report smtrs.pdf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = "./chroma_db_groq"
# --- 2. Document Ingestion ---
print("Loading PDF...")
loader = PyPDFLoader(FILE_PATH)
data = loader.load()

# Chunking helps the retriever find specific sections in your ML notes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(data)

# --- 3. Local Embeddings (Avoids Gemini Quotas) ---
# Using a local HuggingFace model ensures you never hit an embedding rate limit
print("Initializing local embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("Loading existing vector store...")  # ✅ fast
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
else:
    print("Creating vector store for first time...")  # runs once only
    loader = PyPDFLoader(FILE_PATH)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(data)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

# --- 4. Initialize Groq LLM ---
# llama-3.3-70b-versatile provides near-instant reasoning speed
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# --- 5. Build the RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# --- 6. Execution ---
def ask_kb(query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ Show which pages the answer came from
    pages = [str(doc.metadata.get('page', 0) + 1) for doc in docs]
    print(f"Sources: pages {', '.join(pages)}")

    response = llm.invoke(f"""Answer using ONLY this context:
{context}

Question: {query}
If answer not in context say: 'Not found in document'""")

    print(f"Answer: {response.content}")

# Test the system with your Machine Learning content
ask_kb("tell me the summary of this document such that i can understand full report like if someone ask about it i can answer it")
