# ─── Estágio 1: instala dependências ──────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Estágio 2: treina o modelo (usado pelo target "trained") ─────────────────
FROM deps AS trainer

WORKDIR /train
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY src/  ./src/
COPY data/ ./data/

# Se existir um CSV real em data/dataset_real.csv, usa-o misturado com o sintético.
# Caso contrário, usa apenas o dataset sintético gerado automaticamente.
RUN mkdir -p models \
    && python data/synthetic_generator.py \
    && if [ -f data/dataset_real.csv ]; then \
         echo "Dataset real encontrado — convertendo e mesclando..."; \
         python data/converter.py --input data/dataset_real.csv --output data/cold_start_v1.csv; \
       fi \
    && PYTHONPATH=. python src/training.py \
    && echo "Modelo treinado com sucesso."

# ─── Estágio 3a: runtime com modelo pré-treinado (target "trained") ───────────
FROM python:3.11-slim AS trained

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=deps  /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps  /usr/local/bin /usr/local/bin
COPY --from=trainer /train/models/ ./models/
COPY src/ ./src/

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ─── Estágio 3b: runtime sem modelo (target padrão — usa volume) ───────────────
# O modelo é montado em runtime via -v ./models:/app/models
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY src/ ./src/

RUN mkdir -p models && chown -R appuser:appuser /app
USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
