CREATE TABLE documentos (
    id SERIAL PRIMARY KEY,
    sha256 CHAR(64) NOT NULL,
    texto TEXT NOT NULL,
    num_paginas INT NOT NULL,
    formato VARCHAR(20),
    titulo VARCHAR(255),
    autor VARCHAR(255),
    asunto VARCHAR(255),
    palabras_clave TEXT,
    creador VARCHAR(255),
    productor VARCHAR(255),
    fecha_creacion VARCHAR(50),
    fecha_modificacion VARCHAR(50),
    atrapado VARCHAR(50),
    encriptacion BOOLEAN
);
