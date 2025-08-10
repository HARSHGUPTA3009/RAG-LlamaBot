import os
import sys
import traceback
import hashlib
import shutil
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----------------- CONFIG -----------------
DB_DIR = "chroma_db"
KNOWLEDGE_FILE = "knowledge.txt"
HASH_FILE = "knowledge_hash.txt"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3:latest"

# ------------ PROMPT TEMPLATE -------------
STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. ONLY answer the question using the information in the context below. "
        "If the answer is not contained within the context, say 'No relevant information found.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ------------- HELPER FUNCTIONS -----------

def get_file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_knowledge_file(path):
    if not os.path.exists(path):
        print(f"❌ Knowledge file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def build_vectorstore(docs):
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print("✅ ChromaDB built & persisted!")
        return vectorstore
    except Exception as e:
        print(f"❌ ERROR: Failed to build ChromaDB: {e}")
        traceback.print_exc()
        sys.exit(1)

def load_vectorstore():
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"❌ ERROR: Could not load ChromaDB: {e}")
        traceback.print_exc()
        sys.exit(1)

def chat_loop(retriever):
    llm = ChatOllama(model=CHAT_MODEL)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": STRICT_PROMPT},
    )
    print("\n💬 RAG Chat ready! Ask me anything (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            result = qa_chain.invoke({"query": query})
            answer = result['result'] if isinstance(result, dict) else result

            if "No relevant information found." in answer:
                print("⚠️ No relevant info found in context. Querying LLaMA directly...")
                fallback_response = llm.invoke(query)
                print(f"\n🤖 {fallback_response.content.strip()}\n")
            else:
                print(f"\n🤖 {answer.strip()}\n")

        except Exception as e:
            print(f"❌ ERROR during chat: {e}")
            traceback.print_exc()

# --------------- MAIN EXECUTION ------------
if __name__ == "__main__":
    # Compute current knowledge file hash
    current_hash = get_file_hash(KNOWLEDGE_FILE)

    # Read previous hash if exists
    previous_hash = None
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()

    # If hash changed or no DB, rebuild embeddings and vectorstore
    if current_hash != previous_hash or not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("📂 Knowledge file changed or first run, rebuilding vectorstore...")

        # Remove old DB folder if exists to force clean rebuild
        if os.path.exists(DB_DIR):
            print(f"🧹 Removing existing DB folder '{DB_DIR}'...")
            shutil.rmtree(DB_DIR)

        text = load_knowledge_file(KNOWLEDGE_FILE)
        print("✂️ Splitting text into chunks...")
        docs = split_text(text)

        print(f"🧠 Creating embeddings with model '{EMBED_MODEL}'...")
        vectorstore = build_vectorstore(docs)

        # Save the new hash
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
    else:
        print("📦 Loading existing ChromaDB...")
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chat_loop(retriever)
