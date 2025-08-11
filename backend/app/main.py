from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import clasificador_router
from app.api.routers import detectar_route
from app.api.routers import alertas_router

app = FastAPI(
    title="Licitaciones IA API",
    description="API para análisis automatizado de pliegos de licitación",
    version="0.1.0",
)
#app.include_router(extract_pdf_router.router, prefix="/extract")
app.include_router(clasificador_router.router, prefix="/clasificacion")
app.include_router(detectar_route.router, prefix="/deteccion")
app.include_router(alertas_router.router, prefix="/alert")
@app.get("/")
def read_root():
    return {"status": "API funcionando"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (cambia esto en producción)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos los headers
)
