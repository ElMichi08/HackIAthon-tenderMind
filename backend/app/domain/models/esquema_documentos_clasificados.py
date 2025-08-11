from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentoClasificado(Base):
    __tablename__ = "documentos_clasificados"

    id = Column(Integer, primary_key=True)
    ruc_encontrado = Column(String(13), nullable=False)
    razon_social = Column(Text, nullable=False)
    estado_contribuyente_ruc = Column(String(50), nullable=False)
    actividad_economica_principal = Column(Text, nullable=False)
    tipo_contribuyente = Column(String(50), nullable=False)
    regimen = Column(String(50), nullable=False)
    categoria = Column(String(50), nullable=True)
    obligado_llevar_contabilidad = Column(String(2), nullable=False)
    agente_retencion = Column(String(2), nullable=False)
    contribuyente_especial = Column(String(2), nullable=False)
    fecha_inicio_actividades = Column(String(50), nullable=False)
    fecha_cese = Column(String(50), nullable=True)
    fecha_reinicio_actividades = Column(String(50), nullable=True)
    fecha_actualizacion = Column(String(50), nullable=False)
    representante_identificacion = Column(String(20), nullable=False)
    representante_nombre = Column(Text, nullable=False)
    motivo_cancelacion_suspension = Column(Text, nullable=True)
    contribuyente_fantasma = Column(String(2), nullable=False)
    transacciones_inexistente = Column(String(2), nullable=False)
    clasificacion_legal = Column(Text, nullable=True)
    clasificacion_tecnica = Column(Text, nullable=True)
    clasificacion_economica = Column(Text, nullable=True)
    texto_pdf_completo = Column(Text, nullable=False)
    fecha_registro = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP")