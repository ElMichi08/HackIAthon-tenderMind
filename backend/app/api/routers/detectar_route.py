import os
import glob
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# 🧠 Dependencias internas
from app.config.config_db import get_db
from app.domain.models.esquema_documentos_clasificados import DocumentoClasificado

# ===========================
# 🔧 Configuración global
# ===========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PERSIST_DIRECTORY = "./bc_licitaciones"
DOCUMENTOS_ROOT = "./app/documentos"

router = APIRouter(
    tags=["Gestión de documentos y análisis"],
    responses={404: {"description": "No encontrado"}}
)

rag_chain: Optional[RetrievalQA] = None  # Cadena RAG global


# ===========================
# 📦 Funciones base RAG
# ===========================
def cargar_rag(persist_directory=PERSIST_DIRECTORY) -> RetrievalQA:
    """Carga la base vectorial desde disco y devuelve la cadena RAG."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def dividir_documentos(docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """Divide documentos en fragmentos más pequeños."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def detectar_tipo_documento(texto: str) -> str:
    """Detecta tipo de documento basado en el contenido del texto."""
    texto_lower = texto.lower()
    if "contrato" in texto_lower:
        return "contrato"
    elif "pliego" in texto_lower:
        return "pliego"
    elif "propuesta" in texto_lower or "oferta técnica" in texto_lower:
        return "propuesta"
    else:
        return "propuesta"


# ===========================
# 🚀 Endpoints
# ===========================
@router.post("/crear_base_vectorial")
def crear_base_vectorial():
    """Crea y persiste la base vectorial desde PDFs en DOCUMENTOS_ROOT."""
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
            documentos.extend(loader.load())

    if not documentos:
        raise HTTPException(status_code=400, detail="No se encontraron documentos para indexar")

    docs_split = dividir_documentos(documentos)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    global rag_chain
    rag_chain = cargar_rag()

    return {"mensaje": f"Base vectorial creada con {len(docs_split)} fragmentos"}


def validar_documento(texto_completo: str, tipo_documento: str) -> str:
    """Valida un documento usando la cadena RAG global."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Base vectorial no cargada. Ejecuta /crear_base_vectorial primero.")

    prompts = {
        "propuesta": f"""
Analiza la siguiente propuesta técnica/económica:

{texto_completo}

1. ¿Cumple con los requisitos establecidos en los pliegos?
2. ¿Hay vacíos u omisiones relevantes?
3. ¿Cláusulas ambiguas o contradictorias?
""",
        "contrato": f"""
Analiza el siguiente contrato:

{texto_completo}

1. ¿Refleja lo solicitado en los pliegos?
2. ¿Cláusulas contradictorias?
3. ¿Faltan garantías, plazos o penalizaciones?
""",
        "pliego": f"""
Analiza el siguiente pliego:

{texto_completo}

1. ¿Está completo y claro?
2. ¿Ambigüedades que generen problemas en adjudicación?
3. ¿Cláusulas que deben reforzarse?
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
    """Valida todos los documentos en la base de datos."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Base vectorial no cargada. Ejecuta /crear_base_vectorial primero.")

    documentos = db.query(DocumentoClasificado).all()
    resultados = []
    for doc in documentos:
        tipo_doc = detectar_tipo_documento(doc.texto_pdf_completo)
        observaciones = validar_documento(doc.texto_pdf_completo, tipo_doc)
        resultados.append(DocumentoValidado(
            id=doc.id,
            tipo_documento=tipo_doc,
            observaciones=observaciones
        ))
    return resultados


class DocumentoMejoras(BaseModel):
    id: int
    tipo_documento: str
    observaciones: str
    recomendaciones: str
    semaforo_alerta: str


@router.get("/sugerir_mejoras_alertas/{documento_id}", response_model=DocumentoMejoras)
def sugerir_mejoras_alertas(documento_id: int, db: Session = Depends(get_db)):
    """Sugiere mejoras y alerta en semáforo para un documento."""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="Base vectorial no cargada. Ejecuta /crear_base_vectorial primero.")

    doc = db.query(DocumentoClasificado).filter_by(id=documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    tipo_doc = detectar_tipo_documento(doc.texto_pdf_completo)

    prompt = f"""
Eres un experto legal y de contratación pública.
Usa las plantillas oficiales de {tipo_doc} en la base de conocimiento.

Analiza el documento y responde SOLO en formato JSON:
{{
    "observaciones": "...",
    "recomendaciones": "...",
    "semaforo_alerta": "verde|amarillo|rojo"
}}

Documento real:
\"\"\"{doc.texto_pdf_completo}\"\"\"
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
