FROM python:3.11-slim

# Evitar buffers en logs y configurar encoding
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instalar librerías del sistema necesarias para OpenMP y FAISS en Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias primero para aprovechar cache de capas Docker
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y recursos estáticos
COPY . /app/

# Puerto expuesto por FastAPI
EXPOSE 8000

# Verificación de salud del servicio
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -f http://localhost:8000/api/info || exit 1

# Comando de inicio del servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
