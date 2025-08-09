import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()  # carga variables del .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llmEmbedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

chroma_db = Chroma(
    embedding_function=llmEmbedding,
    persist_directory="./bc_licitaciones",
    collection_name="documentos"
)

def save_to_chroma(texto: str, metadata: dict, doc_id: str):
    chroma_db.add_texts(
        texts=[texto],
        metadatas=[metadata],
        ids=[doc_id]
    )
    chroma_db.persist()

def search_in_chroma(query: str):
    retriever = chroma_db.as_retriever()
    return retriever.get_relevant_documents(query)
