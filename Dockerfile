FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema necesarias para Prophet y psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fuente
COPY . .

# Crear directorio para modelos guardados
RUN mkdir -p models

EXPOSE 8001

# Render inyecta PORT dinamicamente — shell form para expandir la variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"]