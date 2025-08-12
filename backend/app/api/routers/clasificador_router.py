from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import tempfile
import requests
import re
import logging
from app.config.config_db import get_db
from app.domain.models.esquema_documentos_clasificados import DocumentoClasificado

load_dotenv()
RUC_API_TOKEN = os.getenv("RUC_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# Prompt para clasificación
prompt_clasificacion = PromptTemplate(
    input_variables=["chunk"],
    template="""
        Eres un experto en licitaciones públicas de Ecuador.
        Clasifica el siguiente texto en una de las siguientes categorías: 

        1. Condiciones legales (garantías, multas, plazos)
        2. Requisitos técnicos (materiales, procesos, tiempos)
        3. Condiciones económicas (presupuestos, formas de pago)

        Texto a clasificar:
        {chunk}

        Devuelve solo el nombre exacto de la categoría.
    """
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt_clasificacion | llm

# Extraer RUC del texto
def extraer_ruc(texto: str):
    match = re.search(r"\b\d{13}\b", texto)
    return match.group(0) if match else None

# Validar RUC vía API
def validar_ruc_api(ruc: str):
    url = f"https://webservices.ec/api/ruc/{ruc}"
    headers = {
        "Authorization": f"Bearer {RUC_API_TOKEN}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        logging.error(f"Error RUC API: {resp.text}")
        raise HTTPException(status_code=502, detail="Error consultando servicio de RUC")

    data = resp.json().get("data", {}).get("main", [])
    if not data:
        raise HTTPException(status_code=404, detail="RUC no encontrado")

    info = data[0]
    if info.get("estadoContribuyenteRuc") != "ACTIVO":
        raise HTTPException(status_code=400, detail="RUC inactivo o suspendido")

    tipo = info.get("tipoContribuyente", "").lower()
    tipos_permitidos = ["sociedad", "compañía", "persona natural"]
    if not any(t in tipo for t in tipos_permitidos):
        raise HTTPException(status_code=400, detail="Tipo de razón social no permitido")

    return info

# Clasificar contenido del PDF
def clasificar_documento(ruta_pdf: str):
    try:
        loader = PyMuPDFLoader(ruta_pdf)
        documentos = loader.load()
    except Exception as e:
        logging.error(f"Error cargando PDF: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar el PDF")

    resultados = {"legal": [], "tecnica": [], "economica": []}
    for doc in documentos:
        try:
            categoria = chain.invoke({"chunk": doc.page_content}).content.lower()
        except Exception as e:
            logging.warning(f"Error clasificando chunk: {e}")
            continue

        if "legal" in categoria:
            resultados["legal"].append(doc.page_content)
        elif "tecnica" in categoria:
            resultados["tecnica"].append(doc.page_content)
        elif "económica" in categoria or "economica" in categoria:
            resultados["economica"].append(doc.page_content)

    return resultados

# Endpoint principal
@router.post("/clasificar/")
async def clasificar_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documentos = loader.load()
        if not documentos:
            raise HTTPException(status_code=400, detail="El PDF no contiene texto legible")

        texto_completo = " ".join([doc.page_content for doc in documentos])
        ruc = extraer_ruc(texto_completo)
        if not ruc:
            raise HTTPException(status_code=400, detail="No se encontró un RUC en el documento")
        print(f"RUC encontrado: {ruc}")

        info_ruc = validar_ruc_api(ruc)
        resultado = clasificar_documento(tmp_path)

        nuevo_doc = DocumentoClasificado(
            ruc_encontrado=ruc,
            razon_social=info_ruc.get("razonSocial"),
            estado_contribuyente_ruc=info_ruc.get("estadoContribuyenteRuc"),
            actividad_economica_principal=info_ruc.get("actividadEconomicaPrincipal"),
            tipo_contribuyente=info_ruc.get("tipoContribuyente"),
            regimen=info_ruc.get("regimen"),
            categoria=info_ruc.get("categoria"),
            obligado_llevar_contabilidad=info_ruc.get("obligadoLlevarContabilidad"),
            agente_retencion=info_ruc.get("agenteRetencion"),
            contribuyente_especial=info_ruc.get("contribuyenteEspecial"),
            fecha_inicio_actividades=info_ruc.get("informacionFechasContribuyente", {}).get("fechaInicioActividades"),
            fecha_cese=info_ruc.get("informacionFechasContribuyente", {}).get("fechaCese"),
            fecha_reinicio_actividades=info_ruc.get("informacionFechasContribuyente", {}).get("fechaReinicioActividades"),
            fecha_actualizacion=info_ruc.get("informacionFechasContribuyente", {}).get("fechaActualizacion"),
            representante_identificacion=info_ruc.get("representantesLegales", [{}])[0].get("identificacion"),
            representante_nombre=info_ruc.get("representantesLegales", [{}])[0].get("nombre"),
            motivo_cancelacion_suspension=info_ruc.get("motivoCancelacionSuspension"),
            contribuyente_fantasma=info_ruc.get("contribuyenteFantasma"),
            transacciones_inexistente=info_ruc.get("transaccionesInexistente"),
            clasificacion_legal="\n\n".join(resultado.get("legal", [])),
            clasificacion_tecnica="\n\n".join(resultado.get("tecnica", [])),
            clasificacion_economica="\n\n".join(resultado.get("economica", [])),
            texto_pdf_completo=texto_completo
        )

        db.add(nuevo_doc)
        db.commit()
        db.refresh(nuevo_doc)

        return JSONResponse(
            content={
                "id": nuevo_doc.id,
                "ruc_encontrado": ruc,
                "ruc_info": info_ruc,
                "clasificacion": resultado,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception as e:
            logging.warning(f"No se pudo eliminar archivo temporal: {e}")
