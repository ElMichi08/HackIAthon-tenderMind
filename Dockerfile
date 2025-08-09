# Imagen base backend
FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app
CMD ["bash", "-lc", "echo 'Use docker-compose.dev.yml para levantar servicios'"]
