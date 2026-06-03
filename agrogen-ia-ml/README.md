<div align="center">

# AgroGen IA — Microsserviço de Inteligência Artificial

**Python 3.11 · FastAPI · scikit-learn · SHAP · K-Means**

Predição de prenhez e análise de padrões reprodutivos para bovinos, ovinos e caprinos.
Hackathon Expoagro Crateús 2026.

</div>

---

## 📦 Tecnologias

| Tecnologia | Versão | Finalidade |
|------------|--------|------------|
| [Python](https://python.org) | 3.11 | Linguagem principal |
| [FastAPI](https://fastapi.tiangolo.com) | 0.110+ | Framework web (REST API) |
| [scikit-learn](https://scikit-learn.org) | 1.4 | Random Forest + K-Means |
| [SHAP](https://shap.readthedocs.io) | 0.45+ | Explicabilidade das predições |
| [pandas](https://pandas.pydata.org) | 2.x | Manipulação de dados |
| [numpy](https://numpy.org) | 1.26+ | Álgebra numérica |
| [joblib](https://joblib.readthedocs.io) | 1.4+ | Serialização do modelo (.pkl) |
| [Pydantic](https://docs.pydantic.dev) | v2 | Validação de schemas |
| [scipy](https://scipy.org) | 1.13+ | Testes estatísticos (KS, qui²) |
| [Uvicorn](https://uvicorn.org) | — | Servidor ASGI |

---

## 🧠 Como a IA funciona

Este módulo implementa uma **arquitetura híbrida de predição**, com dois motores que se complementam:

### Motor 1 — Random Forest (Machine Learning)

O Random Forest é o caminho principal. Funciona como **500 especialistas votando ao mesmo tempo**: cada um analisa os dados do animal a partir de um ângulo diferente e vota "PRENHA" ou "VAZIA". O score final é a proporção de votos positivos.

```
Animal (14 features) ──► Pré-processamento ──► 500 árvores de decisão ──► Score [0.0, 1.0]
                         (encoding + scaler)    (cada uma vota)           (ex: 0.78 = 78%)
```

O modelo foi **treinado** com 1.300 registros sintéticos calibrados por literatura zootécnica (EMBRAPA GENECOC / Hafez 2004) e valida os critérios:

| Métrica | Alvo | Obtido |
|---------|------|--------|
| Acurácia | > 75% | 98.1% |
| AUC-ROC | > 0.80 | 99.8% |
| F1-score (PRENHA) | > 0.78 | 98.5% |
| Latência P95 | ≤ 500ms | ~207ms |

### Motor 2 — Motor de Regras (Fallback determinístico)

Se o modelo ML falhar (arquivo não encontrado, erro de memória, timeout), o motor de regras assume **automaticamente**. Ele não aprende — aplica fórmulas fixas baseadas em zootecnia:

```
score = sigmoid(Σ deltas / 40 + logit(prob_base_espécie))
```

Onde `prob_base` é a taxa histórica por espécie (BOVINO 60%, OVINO/CAPRINO 65%) e os deltas são **26 ajustes parametrizados** por variável (condição corporal, intervalo pós-parto, DEP, temperatura, etc.).

**A resposta sempre informa qual motor foi usado** (`motor_utilizado: "ml"` ou `"rules"`), garantindo transparência total.

### Fluxo completo de uma predição

```
Backend Java
  │
  ├─ POST /predicao { especie, CC, IPP, DEP, temperatura, ... }
  │
  ▼
┌──────────────────────────────────┐
│  src/preprocessing.py            │
│  • Valida ranges (CC ∈ [1,5]...) │
│  • One-hot encoding categóricas  │
│  • MinMaxScaler numéricas        │
└──────────────┬───────────────────┘
               │ array (1, 37 features)
               ▼
┌──────────────────────────────────┐       ┌────────────────────────┐
│  models/rf_v1.0.pkl              │  falha │  src/rules_engine.py   │
│  RandomForest.predict_proba()    │ ──────►│  Σ 26 deltas → sigmoid │
│  → score (ex: 0.78)              │        │  → score (ex: 0.82)    │
└──────────────┬───────────────────┘        └────────────┬───────────┘
               │                                         │
               ▼                                         │
┌──────────────────────────────────┐                     │
│  src/shap_utils.py               │◄────────────────────┘
│  TreeExplainer → top-5 fatores   │  (regras: usa deltas como proxy)
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  src/recommendations.py          │
│  Templates por (feature, sentido)│
│  → 3 recomendações textuais      │
└──────────────┬───────────────────┘
               │
               ▼
  JSON: score, classificação, fatores, recomendações, aviso clínico
```

---

## 🏗️ Estrutura do projeto

```
agrogen-ia-ml/
├── data/
│   ├── synthetic_generator.py   # Gerador do dataset de cold start (1.300 registros)
│   └── cold_start_v1.csv        # Dataset gerado (não versionado)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Análise exploratória dos dados
│   ├── 02_baseline.ipynb        # Treinamento baseline e comparação
│   └── 03_feature_importance.ipynb
│
├── src/
│   ├── serve.py                 # FastAPI: 4 endpoints, auth, lifespan
│   ├── schemas.py               # Pydantic v2: contratos de request/response
│   ├── preprocessing.py         # Preprocessor: fit/transform/validate/save/load
│   ├── training.py              # Pipeline de treino: GridSearchCV + model_card
│   ├── evaluation.py            # Métricas: acurácia, AUC-ROC, F1, anti-viés
│   ├── rules_engine.py          # Motor de regras: 26 deltas + fórmula sigmoide
│   ├── shap_utils.py            # Extração de fatores via TreeExplainer
│   ├── recommendations.py       # Geração de recomendações por templates
│   ├── clustering.py            # K-Means: padrões de fertilidade do rebanho
│   └── monitoring/
│       └── drift_detector.py    # KS + qui-quadrado para detecção de drift
│
├── models/                      # Gerado pelo treinamento — não commitar .pkl
│   ├── rf_v1.0.pkl              # Modelo Random Forest serializado
│   ├── rf_v1.0_preprocessor.pkl # Encoders e scaler salvos
│   └── model_card.json          # Metadados: versão, métricas, features, SHA-256
│
├── tests/
│   ├── test_preprocessing.py    # 15 testes: encoding, scaling, validações
│   ├── test_rules_engine.py     # 27 testes: todos os deltas + Mimosa + extremos
│   └── test_serve.py            # 30 testes: schemas Pydantic + integração HTTP
│
├── conftest.py                  # Configura PYTHONPATH para pytest
├── pytest.ini                   # Configuração do pytest
├── requirements.txt             # Dependências fixadas
├── Dockerfile                   # Build multi-stage, usuário não-root
├── docker-compose.yml           # Serviço único com volume ./models
└── README.md
```

---

## ⚡ Rodando Localmente

### Pré-requisitos

- Python 3.11+
- `pip`

### 1. Instalar dependências

```bash
cd agrogen-ia-ml

# Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
# Crie um arquivo .env ou exporte diretamente:
export BACKEND_AUTH_SECRET=segredo-compartilhado-com-o-backend
export MODEL_PATH=models/rf_v1.0.pkl                    # default
export PREPROCESSOR_PATH=models/rf_v1.0_preprocessor.pkl  # default
export MODEL_CARD_PATH=models/model_card.json             # default
```

### 3. Treinar o modelo

Execute uma vez antes de rodar o servidor pela primeira vez.

#### Com dataset sintético (cold start)

```bash
# Gera o dataset sintético (cold_start_v1.csv) — 1.300 registros calibrados
python data/synthetic_generator.py

# Treina o Random Forest com GridSearchCV (~4 minutos)
# Salva os artefatos em models/
PYTHONPATH=. python src/training.py
```

#### Com seu próprio CSV de inseminações

Se você tiver um CSV com dados reais de inseminações, use o conversor antes de treinar:

```bash
# Converte e mescla com dados sintéticos (garante qualidade do modelo)
python data/converter.py --input /caminho/para/seu_dataset.csv

# Treina normalmente — usa o cold_start_v1.csv gerado pelo converter
PYTHONPATH=. python src/training.py
```

O conversor resolve automaticamente diferenças de formato (maiúsculas, separadores, nomes de colunas) e preenche features ausentes com defaults calibrados. O modelo resultante usa dados reais onde disponíveis e sintéticos para cobrir o restante.

> **Formato mínimo do CSV:** colunas `especie`, `raca_femea`, `condicao_corporal`, `num_partos_anteriores`, `intervalo_pos_parto_dias`, `dias_desde_ultima_ins`, `tipo_inseminacao`, `protocolo_hormonal`, `temperatura_ambiente_c`, `estacao` e `resultado` (`Prenha` ou `Nao_Prenha`). Demais colunas ML ausentes são preenchidas com defaults por espécie.

Artefatos gerados em `models/`:

| Arquivo | Descrição |
|---------|-----------|
| `rf_v1.0.pkl` | Modelo Random Forest treinado e serializado com joblib |
| `rf_v1.0_preprocessor.pkl` | Encoders (one-hot) e scaler (MinMax) salvos do treino |
| `model_card.json` | Versão, métricas, features, hiperparâmetros, SHA-256 do dataset |

### 4. Iniciar o servidor

```bash
PYTHONPATH=. uvicorn src.serve:app --reload --port 8001
# API disponível em http://localhost:8001
```

Logs de startup indicam se o modelo foi carregado:
```
INFO: Modelo rf_v1.0 carregado em memória.
INFO: Uvicorn running on http://0.0.0.0:8001
```

Se o modelo não for encontrado, o servidor sobe mesmo assim — usa o motor de regras como fallback:
```
WARNING: Modelo não encontrado. Execute 'python src/training.py'.
```

---

## 🐳 Rodando com Docker

O Dockerfile tem **duas formas de uso** — escolha a que melhor se adapta ao seu fluxo:

---

### Opção A — Volume externo (padrão, mais rápida)

O modelo é treinado localmente e montado como volume em runtime. O `docker build` é rápido (~2 min) porque não treina.

```bash
# 1. Treine localmente (uma vez)
PYTHONPATH=. python data/synthetic_generator.py
PYTHONPATH=. python src/training.py
# ou com CSV real:
# python data/converter.py --input meu_dataset.csv && PYTHONPATH=. python src/training.py

# 2. Build da imagem (target padrão — sem modelo embutido)
docker build -t agrogen-ia-ml .

# 3. Run montando a pasta models como volume
docker run -d \
  --name agrogen-ia-ml \
  -p 8001:8001 \
  -e BACKEND_AUTH_SECRET=seu-segredo \
  -v $(pwd)/models:/app/models \
  agrogen-ia-ml
```

**Vantagem:** build rápido, fácil de atualizar o modelo sem rebuildar a imagem.

---

### Opção B — Modelo embutido na imagem (recomendado para deploy)

O modelo é **treinado durante o `docker build`** e fica gravado dentro da imagem. Não precisa de volume externo. Ideal para hospedar em Render, Railway, Fly.io etc.

```bash
# Build com modelo sintético embutido (~6 minutos total)
docker build --target trained -t agrogen-ia-ml:trained .

# Run sem volume — modelo já está na imagem
docker run -d \
  --name agrogen-ia-ml \
  -p 8001:8001 \
  -e BACKEND_AUTH_SECRET=seu-segredo \
  agrogen-ia-ml:trained
```

#### Embutir com seu CSV real

```bash
# 1. Copie o CSV para dentro da pasta data/ com o nome dataset_real.csv
cp /caminho/para/seu_dataset.csv data/dataset_real.csv

# 2. Build — o conversor é chamado automaticamente se data/dataset_real.csv existir
docker build --target trained -t agrogen-ia-ml:trained .

# 3. Remova o CSV da pasta após o build (opcional)
rm data/dataset_real.csv
```

---

### Com docker-compose

```bash
# Sobe o serviço (Opção A, com volume)
docker compose up -d

# Verifica os logs
docker compose logs -f agrogen-ia-ml

# Para
docker compose down
```

`docker-compose.yml` já configura o volume `./models:/app/models`, healthcheck e reinício automático.

### Variáveis de ambiente suportadas

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `BACKEND_AUTH_SECRET` | `""` (sem auth em dev) | Token Bearer compartilhado com o backend |
| `MODEL_PATH` | `models/rf_v1.0.pkl` | Caminho para o modelo treinado |
| `PREPROCESSOR_PATH` | `models/rf_v1.0_preprocessor.pkl` | Caminho para o preprocessor |
| `MODEL_CARD_PATH` | `models/model_card.json` | Caminho para os metadados do modelo |

> **Atenção:** Com `BACKEND_AUTH_SECRET` vazio, o servidor aceita qualquer requisição (modo desenvolvimento). Em produção, defina sempre esta variável.

### Healthcheck

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "ok",
  "modelo_versao": "rf_v1.0",
  "modelo_carregado": true,
  "timestamp": "2026-06-05T14:32:00Z"
}
```

---

## 🔌 API

### Autenticação

Todos os endpoints de IA requerem o header:
```
Authorization: Bearer <BACKEND_AUTH_SECRET>
```

`/health` e `/model-info` são públicos (sem auth).

---

### `POST /predicao`

Predição de prenhez para um animal individual.

**Request body:**

```json
{
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

  // Extras usados pelo motor de regras (opcionais, default null/0):
  "dep_fertilidade_reprodutor": 7.5,
  "heterose_esperada": 4.2,
  "ciclos_sem_concepcao": 1
}
```

**Validações automáticas:**

| Campo | Regra |
|-------|-------|
| `condicao_corporal` | ∈ [1.0, 5.0] |
| `historico_taxa_prenhez` | ∈ [0.0, 1.0] |
| `dep_acuracia` | ∈ [0.0, 1.0] |
| `coeficiente_endogamia` | ∈ [0.0, 1.0] |
| `temperatura_ambiente_c` | ≤ 50.0°C |
| `tipo_inseminacao` | `"IATF"` ou `"IA_CONVENCIONAL"` |
| `especie` | `"BOVINO"`, `"OVINO"` ou `"CAPRINO"` |
| `protocolo_hormonal` | Não pode ser `"IA_CONVENCIONAL"` quando `tipo_inseminacao = "IATF"` |

**Response 200:**

```json
{
  "predicao_id": "550e8400-e29b-41d4-a716-446655440000",
  "score_prenhez": 0.78,
  "score_percentual": 78,
  "classificacao": "FAVORAVEL",
  "fatores_determinantes": [
    { "feature": "historico_taxa_prenhez", "valor": 0.75, "impacto": 0.18, "sentido": "positivo" },
    { "feature": "intervalo_pos_parto_dias", "valor": 90,  "impacto": 0.14, "sentido": "positivo" },
    { "feature": "condicao_corporal",       "valor": 4,   "impacto": 0.12, "sentido": "positivo" },
    { "feature": "ciclos_sem_concepcao",    "valor": 1,   "impacto": -0.09, "sentido": "negativo" },
    { "feature": "temperatura_ambiente_c",  "valor": 29,  "impacto": -0.06, "sentido": "negativo" }
  ],
  "recomendacoes": [
    "Histórico individual favorável (75% de prenhez). Animal com bom desempenho reprodutivo.",
    "Atenção à temperatura ambiente (29°C). Realize o procedimento nas horas mais frescas do dia.",
    "Protocolo IATF com sincronização hormonal aumenta a taxa de concepção."
  ],
  "aviso_clinico": "Este score é uma estimativa probabilística baseada em dados históricos e não substitui o julgamento clínico veterinário.",
  "modelo_versao": "rf_v1.0",
  "motor_utilizado": "ml",
  "processado_em_ms": 207
}
```

**Classificação do score:**

| Score | Classificação | Interpretação |
|-------|---------------|---------------|
| ≥ 0.70 | `FAVORAVEL` | Momento adequado para inseminar |
| 0.50–0.69 | `MEDIO` | Há fatores de atenção; revisar antes de prosseguir |
| < 0.50 | `DESFAVORAVEL` | Risco elevado; considerar adiar ou ajustar manejo |

---

### `POST /padroes-fertilidade`

Análise de padrões do rebanho via K-Means. O backend Java envia o dataset completo no body.

**Request body:**

```json
{
  "inseminacoes": [
    {
      "condicao_corporal": 3.8,
      "intervalo_pos_parto_dias": 75,
      "num_partos_anteriores": 3,
      "historico_taxa_prenhez": 0.75,
      "temperatura_ambiente_c": 26,
      "resultado": "PRENHA"
    }
    // ... mínimo 20 registros com diagnóstico confirmado
  ],
  "filtros_aplicados": { "especie": "BOVINO", "periodo": "2026-01 a 2026-06" },
  "min_clusters": 3,
  "max_clusters": 5
}
```

**Response 200:**

```json
{
  "total_inseminacoes_analisadas": 142,
  "clusters": [
    {
      "cluster_id": 0,
      "tamanho": 58,
      "perfil": { "condicao_corporal": 3.8, "intervalo_pos_parto_dias": 82.0, "temperatura_ambiente_c": 26.2 },
      "taxa_prenhez": 0.79,
      "descricao_textual": "Grupo de 58 animais com condição corporal alta (média 3.8), intervalo pós-parto adequado (≥60d) e temperatura amena (26.2°C). Taxa de prenhez: 79%."
    }
  ],
  "insights_principais": [
    "O grupo com condição corporal alta e IPP adequado apresenta taxa 31pp superior ao grupo de pior desempenho.",
    "Animais com condição corporal entre 3 e 4 atingem 79% de prenhez neste rebanho.",
    "Respeitar o intervalo pós-parto mínimo de 60 dias está associado a 75% de sucesso reprodutivo."
  ],
  "metodologia": {
    "algoritmo": "K-Means",
    "n_clusters": 3,
    "features": ["condicao_corporal", "intervalo_pos_parto_dias", "num_partos_anteriores", "historico_taxa_prenhez", "temperatura_ambiente_c"],
    "normalizacao": "StandardScaler"
  }
}
```

---

### `GET /health`

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "ok",
  "modelo_versao": "rf_v1.0",
  "modelo_carregado": true,
  "timestamp": "2026-06-05T14:32:00Z"
}
```

---

### `GET /model-info`

Retorna o conteúdo completo do `model_card.json`.

```bash
curl http://localhost:8001/model-info
```

```json
{
  "modelo_versao": "rf_v1.0",
  "tipo": "RandomForestClassifier",
  "framework": "scikit-learn 1.4.0",
  "treinado_em": "2026-06-05T14:30:00Z",
  "dataset": {
    "n_registros": 1300,
    "split": { "treino": 1040, "teste": 260 },
    "hash_sha256": "606c437b..."
  },
  "hiperparametros": {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 5,
    "class_weight": "balanced"
  },
  "metricas": {
    "acuracia": 0.981,
    "auc_roc": 0.998,
    "f1_prenha": 0.985
  },
  "features_ativas": ["condicao_corporal", "historico_taxa_prenhez", "..."]
}
```

---

## 🤖 As 14 Features do Modelo ML

O Random Forest usa exatamente estas 14 variáveis como entrada:

| Feature | Tipo | Descrição | Faixa |
|---------|------|-----------|-------|
| `condicao_corporal` | Numérica | Escore corporal (1–5) | [1.0, 5.0] |
| `historico_taxa_prenhez` | Numérica | % de prenhezes anteriores | [0.0, 1.0] |
| `intervalo_pos_parto_dias` | Numérica | Dias desde o último parto | ≥ 0 |
| `num_partos_anteriores` | Numérica | Partos confirmados | ≥ 0 |
| `dias_desde_ultima_ins` | Numérica | Dias desde a última inseminação | ≥ 0 |
| `dep_fertilidade_animal` | Numérica | DEP fertilidade da fêmea | livre |
| `dep_acuracia` | Numérica | Confiabilidade dos DEPs | [0.0, 1.0] |
| `coeficiente_endogamia` | Numérica | Coeficiente F da prógenie | [0.0, 1.0] |
| `temperatura_ambiente_c` | Numérica | Temperatura no momento da IA | ≤ 50°C |
| `especie` | Categórica | BOVINO / OVINO / CAPRINO | — |
| `raca_femea` | Categórica | Raça da fêmea | — |
| `tipo_inseminacao` | Categórica | IATF / IA_CONVENCIONAL | — |
| `protocolo_hormonal` | Categórica | Protocolo utilizado | — |
| `estacao` | Categórica | SECA / CHUVOSA | — |

**Extras usados apenas pelo motor de regras** (não entram no ML):

| Campo | Descrição |
|-------|-----------|
| `dep_fertilidade_reprodutor` | DEP fertilidade do reprodutor candidato |
| `heterose_esperada` | Heterose esperada do cruzamento (%) |
| `ciclos_sem_concepcao` | Ciclos consecutivos sem concepção |

---

## 🔄 Pipeline de Treinamento (MLOps)

### Treinamento inicial (cold start)

```bash
# 1. Gerar dataset sintético (1.300 registros calibrados por literatura)
python data/synthetic_generator.py

# 2. Treinar com GridSearchCV — testa 27 combinações de hiperparâmetros
python src/training.py

# 3. (Opcional) Treinar versão diferente do modelo
python src/training.py --version rf_v1.1 --data data/meu_dataset.csv
```

### Grade de hiperparâmetros testados

| Hiperparâmetro | Valores testados |
|----------------|-----------------|
| `n_estimators` | 100, 200, 500 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5, 10 |

5-fold cross-validation, scoring por AUC-ROC. Modelos que não atingem os critérios mínimos são **automaticamente rejeitados**.

### Retreinamento com dados reais

Quando dados reais acumularem, forneça um CSV com as mesmas 14 colunas + coluna `prenha` (0 ou 1):

```bash
python src/training.py --data data/dados_reais_v2.csv --version rf_v2.0
```

Substitua os arquivos em `models/` e reinicie o servidor — o novo modelo é carregado no startup.

### Detecção de drift

```python
from src.monitoring.drift_detector import comparar_distribuicoes, extrair_stats_treino
import pandas as pd

# Extrair estatísticas do treino (salvar junto com o model_card)
df_treino = pd.read_csv("data/cold_start_v1.csv")
stats = extrair_stats_treino(df_treino)

# Comparar com as últimas 200 predições
recentes = [...]  # lista de dicts das features das predições recentes
resultado = comparar_distribuicoes(recentes, stats)

print(resultado["features_com_drift"])  # quantas features em drift
print(resultado["alerta"])              # True se ≥ 3 features com p < 0.05
```

---

## 🧪 Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ -v --cov=src --cov-report=term-missing

# Apenas o motor de regras (cobertura atual: 95%)
pytest tests/test_rules_engine.py -v

# Apenas integração do servidor
pytest tests/test_serve.py -v
```

**Cobertura atual:**

| Módulo | Cobertura |
|--------|-----------|
| `rules_engine.py` | 95% |
| `schemas.py` | 100% |
| `preprocessing.py` | 91% |
| `serve.py` | 89% |
| `shap_utils.py` | 80% |
| `recommendations.py` | 72% |

### Teste de aceitação — Vaca Mimosa

O cenário da seção 5 do documento de especificação é executado automaticamente em `test_rules_engine.py`:

```python
# Mimosa — Nelore, 3 partos, CC=4, IPP=90 dias, taxa=75%, IATF, 29°C
resultado = predict(mimosa_input, aplicar_ruido=False)
assert 0.75 <= resultado["score_prenhez"] <= 0.84
# Σdeltas = 45 → sigmoid(45/40 + logit(0.60)) = 0.8221
```

---

## 🔗 Integração com o Backend Python (FastAPI)

O backend principal chama o microsserviço via HTTP com timeout de 800ms. Exemplo usando `httpx`:

```python
import httpx

IA_SERVICE_URL = "http://localhost:8001"
BACKEND_AUTH_SECRET = "segredo-compartilhado"

async def chamar_microsservico_ia(features: dict) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=0.8) as client:
            resp = await client.post(
                f"{IA_SERVICE_URL}/predicao",
                json=features,
                headers={"Authorization": f"Bearer {BACKEND_AUTH_SECRET}"},
            )
            resp.raise_for_status()
            return resp.json()["data"]
    except (httpx.TimeoutException, httpx.HTTPStatusError):
        return None  # backend aciona seu próprio fallback
```

Em caso de `None` (timeout ou erro), o backend deve acionar o motor de regras local ou retornar erro controlado ao frontend.

---

## 📊 SLA e Performance

| Endpoint | P95 | Observação |
|----------|-----|------------|
| `POST /predicao` (ML + SHAP) | ≤ 500ms | Medido com modelo carregado em memória |
| `POST /predicao` (fallback regras) | ≤ 50ms | Cálculo determinístico puro |
| `POST /padroes-fertilidade` | ≤ 1.500ms | K-Means sobre 20–200 registros |
| `GET /health` | ≤ 10ms | Sem I/O |

O modelo é carregado **uma única vez** no startup e mantido em memória — não há leitura de disco a cada requisição.

---

## 🔧 Scripts úteis

```bash
# Gerar dataset sintético
python data/synthetic_generator.py

# Treinar modelo (com opções)
PYTHONPATH=. python src/training.py
PYTHONPATH=. python src/training.py --version rf_v1.1 --data data/novo.csv

# Rodar servidor em modo desenvolvimento
PYTHONPATH=. uvicorn src.serve:app --reload --port 8001

# Rodar testes com cobertura
PYTHONPATH=. pytest tests/ -v --cov=src

# Verificar se o servidor está respondendo
curl http://localhost:8001/health

# Fazer uma predição de teste (Mimosa)
curl -X POST http://localhost:8001/predicao \
  -H "Authorization: Bearer dev-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "especie": "BOVINO", "raca_femea": "Nelore",
    "condicao_corporal": 4, "num_partos_anteriores": 3,
    "intervalo_pos_parto_dias": 90, "historico_taxa_prenhez": 0.75,
    "dias_desde_ultima_ins": 35, "dep_fertilidade_animal": 8.2,
    "dep_acuracia": 0.72, "coeficiente_endogamia": 0.012,
    "tipo_inseminacao": "IATF", "protocolo_hormonal": "P4+EB 7 dias",
    "temperatura_ambiente_c": 29, "estacao": "SECA",
    "dep_fertilidade_reprodutor": 7.5, "heterose_esperada": 4.2,
    "ciclos_sem_concepcao": 1
  }'
```

---

## 📄 Licença

GNU GPL v3.0 — veja [`../LICENSE`](../LICENSE)
