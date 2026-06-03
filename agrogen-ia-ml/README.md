# AgroGen IA — Microsserviço de IA

Microsserviço Python para predição de prenhez em bovinos, ovinos e caprinos.
Parte do sistema AgroGen IA — Hackathon Expoagro Crateús 2026.

## Stack

- Python 3.11 · FastAPI 0.110 · scikit-learn 1.4 · SHAP 0.45

## Endpoints

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/health` | Não | Status do serviço |
| GET | `/model-info` | Não | Metadados do modelo em produção |
| POST | `/predicao` | Sim | Predição de prenhez (individual) |
| POST | `/padroes-fertilidade` | Sim | Análise de padrões por K-Means |

**Auth**: header `Authorization: Bearer <BACKEND_AUTH_SECRET>`

## Configuração

```bash
cp .env.example .env   # edite BACKEND_AUTH_SECRET
```

## Treinar o modelo (cold start)

```bash
pip install -r requirements.txt
python data/synthetic_generator.py        # gera data/cold_start_v1.csv
python src/training.py                    # treina, avalia e salva em models/
```

O treinamento leva ~4 minutos. Artefatos gerados em `models/`:
- `rf_v1.0.pkl` — modelo Random Forest
- `preprocessor_v1.0.pkl` — encoders e scaler
- `model_card.json` — metadados e métricas

## Rodar localmente

```bash
uvicorn src.serve:app --reload --port 8001
# Variáveis necessárias:
# BACKEND_AUTH_SECRET=<segredo-compartilhado-com-backend>
# MODEL_PATH=models/rf_v1.0.pkl          (default)
# PREPROCESSOR_PATH=models/preprocessor_v1.0.pkl  (default)
# MODEL_CARD_PATH=models/model_card.json  (default)
```

## Rodar com Docker

```bash
# Treine o modelo primeiro (gera models/ localmente)
python data/synthetic_generator.py && python src/training.py

# Build e run
docker build -t agrogen-ia-ml .
docker run -p 8001:8001 \
  -e BACKEND_AUTH_SECRET=seu-segredo \
  -v $(pwd)/models:/app/models \
  agrogen-ia-ml
```

## docker-compose

```bash
docker compose up
```

## Testes

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Exemplo de chamada

```bash
curl -X POST http://localhost:8001/predicao \
  -H "Authorization: Bearer seu-segredo" \
  -H "Content-Type: application/json" \
  -d '{
    "especie": "BOVINO",
    "raca_femea": "Nelore",
    "condicao_corporal": 4,
    "num_partos_anteriores": 3,
    "intervalo_pos_parto_dias": 90,
    "historico_taxa_prenhez": 0.75,
    "dias_desde_ultima_ins": 35,
    "dep_fertilidade_animal": 8.2,
    "dep_acuracia": 0.72,
    "coeficiente_endogamia": 0.012,
    "tipo_inseminacao": "IATF",
    "protocolo_hormonal": "P4+EB 7 dias",
    "temperatura_ambiente_c": 29,
    "estacao": "SECA",
    "dep_fertilidade_reprodutor": 7.5,
    "heterose_esperada": 4.2,
    "ciclos_sem_concepcao": 1
  }'
```

## Limites de SLA

| Endpoint | P95 |
|---|---|
| `/predicao` (ML) | ≤ 500ms |
| `/predicao` (fallback regras) | ≤ 100ms |
| `/padroes-fertilidade` | ≤ 1.500ms |
