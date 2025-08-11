import os
import glob
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Configuración OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PERSIST_DIRECTORY = "./bc_licitaciones"
DOCUMENTOS_ROOT = "./app/documentos"

rag_chain = None  # cache global para evitar recargar

def cargar_rag(persist_directory: str = PERSIST_DIRECTORY):
    """Carga la base vectorial existente."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def inicializar_rag():
    """Inicializa rag_chain global si no existe."""
    global rag_chain
    if rag_chain is None:
        rag_chain = cargar_rag()
    return rag_chain

def crear_base_vectorial():
    """Escanea documentos PDF y construye la base vectorial."""
    tipos = ["pliego", "propuesta", "contrato"]
    documentos = []

    for tipo in tipos:
        carpeta = os.path.join(DOCUMENTOS_ROOT, tipo)
        if not os.path.exists(carpeta):
            print(f"No existe la carpeta: {carpeta}")
            continue

        pdf_files = glob.glob(os.path.join(carpeta, "**", "*.pdf"), recursive=True)
        pdf_files += glob.glob(os.path.join(carpeta, "**", "*.PDF"), recursive=True)

        print(f"Archivos PDF encontrados en {carpeta}: {pdf_files}")

        for pdf_path in pdf_files:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            documentos.extend(docs)

    if not documentos:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    inicializar_rag()
    return len(docs_split)
