FROM python:3.12.7-slim

WORKDIR /MLDocker

# Préparer les fichiers à copier
COPY requirements.txt .
COPY dvc.yaml dvc.lock data.dvc ./
COPY .dvc/ .dvc/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY config.yaml ./

# Installer les dépendances système + DVC
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git gcc g++ curl ca-certificates libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install dvc[s3] && \
    rm -rf /var/lib/apt/lists/*

# Init Git pour DVC
RUN git init

# Télécharger les données
RUN dvc pull -v --force

# PORT Streamlit
EXPOSE 8501

# Lancement de l'app
CMD ["sh", "-c", "dvc repro && PYTHONPATH=. streamlit run /app/streamlit_app.py"]
