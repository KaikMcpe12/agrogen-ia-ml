FROM python:3.11-slim

WORKDIR /app

# Instala dependências primeiro (cache de layer — só reinstala se requirements mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código e os dados
COPY src/  ./src/
COPY data/ ./data/

# Treina o modelo durante o build (~4 minutos)
# Os artefatos ficam em models/ dentro da imagem — sem volume externo necessário
RUN mkdir -p models \
    && python data/synthetic_generator.py \
    && PYTHONPATH=. python src/training.py

# Expose não é obrigatório no Render, mas documenta a porta padrão
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Render injeta $PORT automaticamente — usamos shell form para expandir a variável
CMD uvicorn src.serve:app --host 0.0.0.0 --port ${PORT:-8001}
