FROM python:3.10-slim

# 1. Instalar dependencias necesarias para OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Crear directorio de trabajo
WORKDIR /app

# 3. Copiar dependencias y código fuente
COPY requirements.txt .
COPY . .
COPY models/ ./models/

# 4. Verificar que los modelos están copiados
RUN ls -l ./models

# 5. Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. Dar permisos al script de inicio
RUN chmod +x start.sh

# 7. Exponer puerto
EXPOSE 80

# 8. Define el ENTRYPOINT para tu aplicación
# ¡ESTO ES LO CRUCIAL! Le dice a Docker que este es el ejecutable principal.
# Reemplaza la línea 'CMD' anterior por esta 'ENTRYPOINT'.
ENTRYPOINT ["/app/start.sh"]

