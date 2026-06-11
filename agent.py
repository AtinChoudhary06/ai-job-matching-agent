import os
import hashlib
from dotenv import load_dotenv

# Modern LangChain imports (works with langchain >= 0.2)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# ──────────────────────────────────────────────
# 1. RESUME LOADER
# ──────────────────────────────────────────────

def load_resume(pdf_path: str) -> Chroma:
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    collection_name = f"resume_{file_hash}"
    persist_dir = "./chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    if vectorstore._collection.count() > 0:
        print(f"[INFO] Resume already indexed. Skipping.")
        return vectorstore

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    print(f"[INFO] Indexing {len(chunks)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return vectorstore


# ──────────────────────────────────────────────
# 2. PROMPT
# ──────────────────────────────────────────────

MATCH_PROMPT = PromptTemplate.from_template("""
You are an expert career advisor.

Resume content:
{context}

Job Description:
{question}

Analyze and return exactly this format:

## Match Score
Give a score out of 100 with a one-line reason.

## Matching Skills
Bullet points of skills in the resume that match the job.

## Missing Skills
Bullet points of skills in the job description NOT found in the resume.

## Recommendations
3 specific actions the candidate should take to improve their chances.

## Verdict
One sentence: apply now or upskill first?
""")


# ──────────────────────────────────────────────
# 3. BUILD AGENT (modern LCEL chain)
# ──────────────────────────────────────────────

def build_agent(vectorstore: Chroma):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
        max_tokens=1500,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain — no RetrievalQA needed
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | MATCH_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ──────────────────────────────────────────────
# 4. MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────

def analyze_match(pdf_path: str, job_description: str) -> str:
    vectorstore = load_resume(pdf_path)
    chain = build_agent(vectorstore)
    result = chain.invoke(job_description)
    return result


# ──────────────────────────────────────────────
# 5. TERMINAL TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py path/to/resume.pdf")
    else:
        sample_jd = """
        Hiring a Junior Data Scientist:
        - Python, Pandas, NumPy
        - Machine Learning: XGBoost, Scikit-learn
        - NLP or LLMs experience is a plus
        - Streamlit or FastAPI for deployment
        - SQL databases
        - Fresher or 0-2 years experience
        """
        print("Analyzing...\n")
        output = analyze_match(sys.argv[1], sample_jd)
        print(output)