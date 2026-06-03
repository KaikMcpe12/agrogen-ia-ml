# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgroGen IA is a genetic and reproductive management system for cattle, sheep, and goats, built for Hackathon Expoagro Crateús 2026. It consists of two services:

- **Backend principal**: Python 3.11 + FastAPI + PostgreSQL 15
- **Microsserviço de IA**: Python 3.11 + FastAPI + scikit-learn

## Tech Stack & Commands

### Backend principal
```bash
pip install -r requirements.txt
uvicorn main:app --reload       # Run dev server (port 8000)
pytest                          # Run all tests
pytest tests/test_auth.py       # Run single test file
```

### Microsserviço de IA
```bash
pip install -r requirements.txt
uvicorn main:app --reload       # Run dev server (port 8001)
pytest                          # Run all tests
pytest tests/test_model.py      # Run single test file
```

### Database
PostgreSQL 15. Migrations via Alembic. Never run raw DDL manually.

## Architecture

### Two-service design
The FastAPI backend handles all business logic, auth, and data persistence. The ML microservice is called synchronously via REST with a **800ms timeout** for ML inference only. If the ML call fails or times out, the backend falls back to the Python rules engine automatically — the system must always return a response.

```
React client
  └── POST /api/v1/ia/predicao-prenhez
        └── Backend FastAPI (auth + feature assembly + PostgreSQL)
              ├── [success] ML microservice FastAPI (Random Forest ~180ms)
              └── [fail/timeout] rules engine (Python, inside backend, <50ms)
                    └── Post-processing: add clinical disclaimer, log to tb_predicao_log
```

### AI hybrid approach
- **Random Forest** (Python/scikit-learn): pregnancy prediction — primary path
- **Motor de Regras** (Python, inside backend): deterministic fallback, always available
- **K-Means** (Python/scikit-learn): fertility pattern clustering — separate endpoint

Every prediction response **must** include top-5 feature importances and a mandatory clinical disclaimer injected by the backend (never suppressed by frontend).

### API conventions
- All endpoints are prefixed `/api/v1/`
- All non-public endpoints require `Authorization: Bearer <access_token>`
- All listing endpoints require `fazenda_id` as query param or `X-Fazenda-ID` header
- Error responses use internal codes (e.g., `AUTH_INVALID_CREDENTIALS`, `VALIDATION_ERROR`)
- `access_token` TTL: 24h; `refresh_token` TTL: 7d (rotation on each refresh)
- Account lockout: 5 consecutive failures → locked for 15 minutes

## Domain Model (15 entities, 5 groups)

**Group 1 — Identity**: `Usuario` (USR), `Fazenda` (FAZ)
**Group 2 — Rebanho**: `Animal` (ANI), `DadosGeneticos` (GEN), `Reprodutor` (REP)
**Group 3 — Reprodução**: `EventoInseminacao`, `DiagnosticoGestacao`, `ProtocoloHormonal`, `Parto`
**Group 4 — Diário de Bordo**: `Pesagem`, `EventoSanitario`, `Alimentacao`, `Ocorrencia`
**Group 5 — Sistema**: `Alerta`, `LogAuditoria`

`Animal` is the central entity — every event (insemination, weighing, health, birth, AI predictions) is a direct or indirect child. `Fazenda` is the administrative scope — all data is segmented by farm.

## Key Design Invariants

- **Soft delete everywhere**: never use SQL `DELETE`. Use `ativo = false`. Inactive records preserve history.
- **UUIDs as PKs**: random UUID v4. Never expose raw UUIDs to end users in URLs where a human-readable code (`codigo`) exists.
- **RBAC roles**: `ADMIN | PRODUTOR | TECNICO | VETERINARIO` — enforced on every protected endpoint.
- **`senha_hash`** (BCrypt cost 12): never returned by any API response.
- **`cpf`**: sensitive field (LGPD art. 18) — mask in all profile displays.
- `Animal.codigo` format: `BOV-0001`, `OVI-0001`, `CAP-0001` — unique per farm.
- `Animal.num_partos` and `data_ultimo_parto` are computed by trigger/JPA listener on confirmed `Parto`, not updated manually.
- `coeficiente_endogamia > 0.0625` triggers an automatic consanguinity alert.

## API Modules

| Prefix | Description |
|---|---|
| `/api/v1/auth` | Login, registro, refresh, logout, recuperar-senha |
| `/api/v1/usuarios` | Perfil, LGPD data export |
| `/api/v1/fazendas` | CRUD propriedades rurais |
| `/api/v1/animais` | CRUD rebanho, importação CSV |
| `/api/v1/reprodutores` | CRUD machos e sêmen externo |
| `/api/v1/inseminacoes` | Eventos de IA, diagnósticos de gestação |
| `/api/v1/diario` | Pesagens, partos, sanidade, ocorrências, export PDF |
| `/api/v1/ia` | Predição de prenhez, padrões de fertilidade, seleção genética |
| `/api/v1/alertas` | Listagem, marcação de lidos, resolução |
| `/api/v1/dashboard` | KPIs, gráficos, timeline |
| `/api/v1/relatorios` | Geração e exportação CSV/PDF |
| `/api/v1/referencia` | Raças por espécie, protocolos hormonais |

## ML Model Targets

| Metric | Target |
|---|---|
| Accuracy | > 75% |
| AUC-ROC | > 0.80 |
| F1-score (PRENHA class) | > 0.78 |
| Inference latency | ≤ 1.000ms (P95), internal goal 500ms |

Model is retrained every 3 months or 500 new labeled records. Cold-start dataset: 1,300 synthetic records calibrated from EMBRAPA GENECOC / Hafez 2004. Serialization format: joblib (`.pkl`). Datasets stored as versioned CSVs in a separate auxiliary repo — not in the operational database.
