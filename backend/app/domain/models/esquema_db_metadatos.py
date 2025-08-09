from sqlalchemy import Column, Integer, String, Text, Boolean, TIMESTAMP, CHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Documento(Base):
    __tablename__ = "documentos"

    id = Column(Integer, primary_key=True, index=True)
    sha256 = Column(CHAR(64), nullable=False, unique=True, index=True)
    texto = Column(Text, nullable=False)
    num_paginas = Column(Integer, nullable=False)
    formato = Column(String(20))
    titulo = Column(String(255))
    autor = Column(String(255))
    asunto = Column(String(255))
    palabras_clave = Column(Text)
    creador = Column(String(255))
    productor = Column(String(255))
    fecha_creacion = Column(TIMESTAMP)
    fecha_modificacion = Column(TIMESTAMP)
    atrapado = Column(String(50))
    encriptacion = Column(Boolean)
