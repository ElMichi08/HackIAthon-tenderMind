import os
import glob
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from app.config.config_db import get_db
from app.domain.models.esquema_documentos_clasificados import DocumentoClasificado

# Configuración OpenAI y variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

router = APIRouter(
    tags=["detección y validación de documentos"],
    responses={404: {"description": "No encontrado"}}
)

PERSIST_DIRECTORY = "./bc_licitaciones"

DOCUMENTOS_ROOT = "./app/documentos"

def cargar_rag(persist_directory=PERSIST_DIRECTORY):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

rag_chain = None  # Se inicializa después de construir la base vectorial

@router.post("/crear_base_vectorial")
def crear_base_vectorial():
    tipos = ["pliego", "propuesta", "contrato"]
    documentos = []

    for tipo in tipos:
        carpeta = os.path.join(DOCUMENTOS_ROOT, tipo)
        if not os.path.exists(carpeta):
            print(f"No existe la carpeta: {carpeta}")
            continue

        # Buscar PDFs recursivamente incluyendo subcarpetas
        pdf_files = glob.glob(os.path.join(carpeta, "**", "*.pdf"), recursive=True)
        pdf_files += glob.glob(os.path.join(carpeta, "**", "*.PDF"), recursive=True)

        print(f"Archivos PDF encontrados en {carpeta}: {pdf_files}")

        for pdf_path in pdf_files:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            documentos.extend(docs)

    if not documentos:
        raise HTTPException(status_code=400, detail="No se encontraron documentos para indexar")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    global rag_chain
    rag_chain = cargar_rag()

    return {"mensaje": f"Base vectorial creada con {len(docs_split)} fragmentos de documentos"}

def detectar_tipo_documento(texto: str) -> str:
    texto_lower = texto.lower()
    if "contrato" in texto_lower:
        return "contrato"
    elif "pliego" in texto_lower:
        return "pliego"
    elif "propuesta" in texto_lower or "oferta técnica" in texto_lower:
        return "propuesta"
    else:
        return "propuesta"

def validar_documento(texto_completo: str, tipo_documento: str = "propuesta") -> str:
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="La base vectorial no está cargada. Ejecuta /crear_base_vectorial primero.")
     # 1️⃣ Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents([Document(page_content=texto_completo)])

    # 2️⃣ Crear embeddings temporales para este texto
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    temp_vectordb = Chroma.from_documents(docs_split, embeddings)

    # 3️⃣ Crear un retriever temporal para esta validación
    temp_retriever = temp_vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    temp_rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=temp_retriever)

    prompts = {
        "propuesta": f"""
Analiza el siguiente texto de una propuesta técnica o económica:

{texto_completo}

1. ¿Cumple con los requisitos establecidos en los pliegos?
2. ¿Hay vacíos, omisiones o incumplimientos relevantes?
3. ¿Qué cláusulas parecen ambiguas o contradictorias?
Responde con una lista clara y justificada.
""",
        "contrato": f"""
Analiza el siguiente contrato:

{texto_completo}

1. ¿Refleja fielmente lo establecido en los pliegos?
2. ¿Hay cláusulas que se contradicen con lo solicitado?
3. ¿Faltan garantías, plazos, penalizaciones o condiciones clave?
Responde con observaciones detalladas.
""",
        "pliego": f"""
Analiza el siguiente pliego:

{texto_completo}

1. ¿Está completo y claro en sus requerimientos?
2. ¿Hay ambigüedades que podrían generar problemas en la adjudicación?
3. ¿Qué cláusulas deberían reforzarse?
Responde con recomendaciones.
"""
    }

    pregunta = prompts.get(tipo_documento, f"Analiza este documento:\n{texto_completo}")
    return rag_chain.run(pregunta)

class DocumentoValidado(BaseModel):
    id: int
    tipo_documento: str
    observaciones: str

@router.get("/validar_documentos", response_model=List[DocumentoValidado])
def validar_desde_bdd(db: Session = Depends(get_db)):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="La base vectorial no está cargada. Ejecuta /crear_base_vectorial primero.")

    documentos = db.query(DocumentoClasificado).all()
    resultados = []
    for doc in documentos:
        tipo_doc = detectar_tipo_documento(doc.texto_pdf_completo)
        observaciones = validar_documento(doc.texto_pdf_completo, tipo_doc)
        resultados.append(DocumentoValidado(id=doc.id, tipo_documento=tipo_doc, observaciones=observaciones))
    return resultados
