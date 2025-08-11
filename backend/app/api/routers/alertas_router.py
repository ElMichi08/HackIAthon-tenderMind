import os
import glob
import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# 🧠 Dependencias internas
from app.config.config_db import get_db
from app.domain.models.esquema_documentos_clasificados import DocumentoClasificado

# Configuración OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

router = APIRouter(
    tags=["Alertas y sugerencias de mejora"],
    responses={404: {"description": "No encontrado"}}
)

PERSIST_DIRECTORY = "./bc_licitaciones"
DOCUMENTOS_ROOT = "./app/documentos"

# ===== Funciones base RAG =====
def cargar_rag(persist_directory=PERSIST_DIRECTORY):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

rag_chain = None  # Se inicializa luego

@router.post("/crear_base_vectorial")
def crear_base_vectorial():
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
        raise HTTPException(status_code=400, detail="No se encontraron documentos para indexar")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(documentos)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    global rag_chain
    rag_chain = cargar_rag()

    return {"mensaje": f"Base vectorial creada con {len(docs_split)} fragmentos"}

# ===== Funciones auxiliares =====
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

# ===== Modelos Pydantic =====
class DocumentoMejoras(BaseModel):
    id: int
    tipo_documento: str
    observaciones: str
    recomendaciones: str
    semaforo_alerta: str

# ===== Nuevo endpoint fusionado =====
@router.get("/sugerir_mejoras_alertas/{documento_id}", response_model=DocumentoMejoras)
def sugerir_mejoras_alertas(documento_id: int, db: Session = Depends(get_db)):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Base vectorial no está cargada. Ejecuta /crear_base_vectorial primero.")

    # Obtener documento desde la BDD
    doc = db.query(DocumentoClasificado).filter_by(id=documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    tipo_doc = detectar_tipo_documento(doc.texto_pdf_completo)

    prompt = f"""
Eres un experto legal y de contratación pública.  
Tienes acceso a PLANTILLAS oficiales de {tipo_doc} en la base de conocimiento.

Analiza este documento real y compáralo con las plantillas para identificar:

1. Cláusulas faltantes o redactadas de forma riesgosa.
2. Recomendaciones de mejora.
3. Nivel de alerta usando un semáforo:
   - verde = sin riesgos importantes.
   - amarillo = riesgos moderados.
   - rojo = riesgos críticos.

Documento real:
\"\"\"{doc.texto_pdf_completo}\"\"\"


Responde ÚNICAMENTE en formato JSON:
{{
    "observaciones": "...",
    "recomendaciones": "...",
    "semaforo_alerta": "verde|amarillo|rojo"
}}
"""

    respuesta = rag_chain.run(prompt)

    try:
        datos = json.loads(respuesta)
        observaciones = datos.get("observaciones", "")
        recomendaciones = datos.get("recomendaciones", "")
        semaforo_alerta = datos.get("semaforo_alerta", "amarillo")
    except Exception:
        observaciones = respuesta
        recomendaciones = ""
        semaforo_alerta = "amarillo"

    return DocumentoMejoras(
        id=doc.id,
        tipo_documento=tipo_doc,
        observaciones=observaciones,
        recomendaciones=recomendaciones,
        semaforo_alerta=semaforo_alerta
    )