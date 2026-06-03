"""
FastAPI — Microsserviço de IA do AgroGen IA.

Endpoints:
  GET  /health               — status do serviço (sem auth)
  GET  /model-info           — metadados do modelo (sem auth)
  POST /predicao             — predição de prenhez individual (requer auth)
  POST /padroes-fertilidade  — análise K-Means do rebanho (requer auth)

Autenticação: Bearer estático via variável BACKEND_AUTH_SECRET.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src import recommendations, rules_engine, shap_utils
from src.clustering import analisar_padroes
from src.preprocessing import FEATURES_ML, Preprocessor
from src.schemas import (
    FatorDeterminante,
    HealthResponse,
    ModelInfoResponse,
    PadroesRequest,
    PadroesResponse,
    PredicaoRequest,
    PredicaoResponse,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "rf_v1.0.pkl")))
PREPROCESSOR_PATH = Path(
    os.environ.get("PREPROCESSOR_PATH", str(BASE_DIR / "models" / "rf_v1.0_preprocessor.pkl"))
)
MODEL_CARD_PATH = Path(
    os.environ.get("MODEL_CARD_PATH", str(BASE_DIR / "models" / "model_card.json"))
)
BACKEND_AUTH_SECRET: str = os.environ.get("BACKEND_AUTH_SECRET", "")

AVISO_CLINICO = (
    "Este score é uma estimativa probabilística baseada em dados históricos "
    "e não substitui o julgamento clínico veterinário."
)

# ---------------------------------------------------------------------------
# Estado global (carregado no startup)
# ---------------------------------------------------------------------------

_state: dict = {
    "model": None,
    "preprocessor": None,
    "explainer": None,
    "model_card": None,
    "modelo_versao": "desconhecida",
    "carregado": False,
}

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e preprocessor em memória ao iniciar o servidor."""
    try:
        _state["model"] = joblib.load(MODEL_PATH)
        _state["preprocessor"] = Preprocessor.load(PREPROCESSOR_PATH)
        _state["explainer"] = shap_utils.criar_explainer(_state["model"])

        if MODEL_CARD_PATH.exists():
            _state["model_card"] = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
            _state["modelo_versao"] = _state["model_card"].get("modelo_versao", "rf_v1.0")

        _state["carregado"] = True
        logger.info("Modelo %s carregado em memória.", _state["modelo_versao"])
    except FileNotFoundError as exc:
        logger.warning(
            "Modelo não encontrado: %s. Inicie com 'python src/training.py'.", exc
        )
    yield


app = FastAPI(
    title="AgroGen IA — Microsserviço ML",
    description="Predição de prenhez por Random Forest + fallback motor de regras.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

_http_bearer = HTTPBearer(auto_error=False)


def verificar_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_http_bearer),
) -> None:
    """Valida Bearer token estático. Pula validação se BACKEND_AUTH_SECRET não configurado."""
    if not BACKEND_AUTH_SECRET:
        return  # modo dev: sem auth
    if credentials is None or credentials.credentials != BACKEND_AUTH_SECRET:
        raise HTTPException(
            status_code=401,
            detail={"codigo": "AUTH_INVALID", "mensagem": "Token de autorização inválido ou ausente."},
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["carregado"] else "degraded",
        modelo_versao=_state["modelo_versao"] if _state["carregado"] else None,
        modelo_carregado=_state["carregado"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Sistema"])
async def model_info() -> ModelInfoResponse:
    if not _state["carregado"] or _state["model_card"] is None:
        raise HTTPException(
            status_code=503,
            detail={"codigo": "MODEL_NOT_LOADED", "mensagem": "Modelo não carregado."},
        )
    return ModelInfoResponse(**_state["model_card"])


# ---------------------------------------------------------------------------
# POST /predicao
# ---------------------------------------------------------------------------


@app.post("/predicao", response_model=PredicaoResponse, tags=["IA"])
async def predicao(
    req: PredicaoRequest,
    _: None = Depends(verificar_auth),
) -> PredicaoResponse:
    """
    Predição de prenhez para um animal.

    Tenta o modelo Random Forest. Se falhar (modelo não carregado ou erro
    de inferência), aciona automaticamente o motor de regras determinístico.
    """
    inicio = time.perf_counter()
    input_dict = req.model_dump()

    motor: str
    score: float
    fatores: list[FatorDeterminante]

    # ------ Caminho ML ------
    if _state["carregado"]:
        try:
            score, fatores, motor = _predizer_ml(input_dict)
        except Exception as exc:
            logger.warning("Falha no caminho ML, ativando fallback: %s", exc)
            score, fatores, motor = _predizer_regras(input_dict)
    else:
        score, fatores, motor = _predizer_regras(input_dict)

    classificacao = _classificar(score)
    recomendacoes = recommendations.gerar_recomendacoes(fatores, input_dict, classificacao)
    ms = int((time.perf_counter() - inicio) * 1000)

    return PredicaoResponse(
        predicao_id=str(uuid.uuid4()),
        score_prenhez=round(score, 4),
        score_percentual=round(score * 100),
        classificacao=classificacao,
        fatores_determinantes=fatores,
        recomendacoes=recomendacoes,
        aviso_clinico=AVISO_CLINICO,
        modelo_versao=_state["modelo_versao"],
        motor_utilizado=motor,
        processado_em_ms=ms,
    )


def _predizer_ml(input_dict: dict) -> tuple[float, list[FatorDeterminante], str]:
    """Inferência via Random Forest + SHAP."""
    prep: Preprocessor = _state["preprocessor"]
    model = _state["model"]
    explainer = _state["explainer"]

    # Valida range antes de transformar
    prep.validate(input_dict)

    # Filtra apenas as 14 features ML (descarta extras do motor de regras)
    ml_input = {k: input_dict.get(k) for k in FEATURES_ML}
    X = prep.transform(ml_input)

    proba = model.predict_proba(X)[0][1]  # P(PRENHA)
    score = float(np.clip(proba, 0.01, 0.99))

    fatores = shap_utils.extrair_fatores(
        explainer=explainer,
        X_transformed=X,
        feature_names_out=prep.feature_names_out,
        valores_originais=input_dict,
        model=model,
        top_k=5,
    )

    return score, fatores, "ml"


def _predizer_regras(input_dict: dict) -> tuple[float, list[FatorDeterminante], str]:
    """Inferência pelo motor de regras determinístico."""
    resultado = rules_engine.predict(input_dict, top_k=5, aplicar_ruido=True)

    score = float(resultado["score_prenhez"])
    fatores_raw = resultado["fatores_determinantes"]

    fatores = [
        FatorDeterminante(
            feature=f["feature"],
            valor=f["valor"],
            impacto=f["impacto"],
            sentido=f["sentido"],
            label=f.get("label"),
        )
        for f in fatores_raw
    ]

    return score, fatores, "rules"


def _classificar(score: float) -> str:
    if score >= 0.70:
        return "FAVORAVEL"
    if score >= 0.50:
        return "MEDIO"
    return "DESFAVORAVEL"


# ---------------------------------------------------------------------------
# POST /padroes-fertilidade
# ---------------------------------------------------------------------------


@app.post("/padroes-fertilidade", response_model=PadroesResponse, tags=["IA"])
async def padroes_fertilidade(
    req: PadroesRequest,
    _: None = Depends(verificar_auth),
) -> PadroesResponse:
    """
    Análise de padrões de fertilidade do rebanho via K-Means.
    O backend Java envia o dataset completo no body; o microsserviço processa e
    devolve clusters + insights textuais gerados por templates.
    """
    return analisar_padroes(req)


# ---------------------------------------------------------------------------
# Entrypoint direto (dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8001, reload=True)
