from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import PDF_PATH, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def ingest_pdf():
    print("[1/3] Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("[2/3] Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks")

    print("[3/3] Storing embeddings in ChromaDB...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Ingestion complete. {len(chunks)} chunks stored.\n")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf()