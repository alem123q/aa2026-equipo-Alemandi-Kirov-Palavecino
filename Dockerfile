# Entorno de la cátedra en Docker — Aprendizaje Automático y Grandes Datos, UNRaf 2026
#
# Construir:   docker build -t unraf-aa .
# Ejecutar:    docker run --rm -p 8888:8888 -v "%cd%":/trabajo unraf-aa      (CMD de Windows)
#              docker run --rm -p 8888:8888 -v ${PWD}:/trabajo unraf-aa      (PowerShell)
# Después:     abrir http://localhost:8888 en el navegador.
#
# Sin contraseña ni token: el servidor solo escucha en la propia computadora.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATOS_CATEDRA=/trabajo/datos/cache

WORKDIR /trabajo

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/trabajo"]
