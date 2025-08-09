from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import magic
import hashlib
from app.domain.services import pdf_sparser_service, chromadb_service
from sqlalchemy.orm import Session
from app.domain.models.esquema_db_metadatos import Documento

router = APIRouter()
from app.config.config_db import get_db


@router.post("/procesar")
async def process_pdf(file: UploadFile, db: Session = Depends(get_db)):
    file_content = await file.read()

    # Validar PDF
    mime = magic.from_buffer(file_content, mime=True)
    if mime != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo no es un PDF válido.")

    # Extraer texto y metadatos
    pdf_data = pdf_sparser_service.extract_pdf_metadata(file_content)
    texto = pdf_data["text"]
    if not texto.strip():
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

    # Hash único
    sha256 = hashlib.sha256(file_content).hexdigest()
    existe = db.query(Documento).filter(Documento.sha256 == sha256).first()
    if existe:
        return JSONResponse({"mensaje": "Documento ya existe", "sha256": sha256})

    metadata = pdf_data["metadata"]

    nuevo_documento = Documento(
        sha256=sha256,
        texto=texto,
        num_paginas=pdf_data["num_pages"],
        formato=metadata.get("format"),
        titulo=metadata.get("title"),
        autor=metadata.get("author"),
        asunto=metadata.get("subject"),
        palabras_clave=metadata.get("keywords"),
        creador=metadata.get("creator"),
        productor=metadata.get("producer"),
        fecha_creacion=metadata.get("creationDate"),
        fecha_modificacion=metadata.get("modDate"),
        atrapado=metadata.get("trapped"),
        encriptacion=metadata.get("encrypted"),
    )

    db.add(nuevo_documento)
    db.commit()
    db.refresh(nuevo_documento)

    # Guardar en ChromaDB

    return JSONResponse(
        {
            "mensaje": "Documento procesado y guardado en ChromaDB",
            "sha256": sha256,
            "text": texto,
            "num_pages": pdf_data["num_pages"],
            "metadata": pdf_data["metadata"],
        }
    )
