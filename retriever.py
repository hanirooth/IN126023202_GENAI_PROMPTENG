from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, EMBEDDING_MODEL, RETRIEVAL_K

_embedding_fn = None
_vectorstore = None

def get_vectorstore():
    global _embedding_fn, _vectorstore
    if _vectorstore is None:
        print("Loading embedding model (one-time)...")
        _embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_embedding_fn
        )
        print(f"ChromaDB loaded. Total chunks: {_vectorstore._collection.count()}")
    return _vectorstore

def retrieve_docs(query: str):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    print(f"Retrieved {len(docs)} docs for query: '{query}'")   # debug line
    return docs