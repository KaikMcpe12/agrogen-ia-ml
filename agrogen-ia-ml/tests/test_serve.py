"""
Testes de schemas Pydantic e integração do servidor FastAPI.

Cobre: validação de request/response, fallback ML→regras,
endpoints /health e /model-info.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.schemas import (
    FatorDeterminante,
    PadroesRequest,
    PredicaoRequest,
    PredicaoResponse,
)

# ---------------------------------------------------------------------------
# Fixture: payload válido da Mimosa
# ---------------------------------------------------------------------------

MIMOSA_PAYLOAD = {
    "especie": "BOVINO",
    "raca_femea": "Nelore",
    "condicao_corporal": 4.0,
    "historico_taxa_prenhez": 0.75,
    "intervalo_pos_parto_dias": 90,
    "num_partos_anteriores": 3,
    "dias_desde_ultima_ins": 35,
    "dep_fertilidade_animal": 8.2,
    "tipo_inseminacao": "IATF",
    "protocolo_hormonal": "P4+EB 7 dias",
    "temperatura_ambiente_c": 29.0,
    "estacao": "SECA",
    "dep_acuracia": 0.80,
    "coeficiente_endogamia": 0.012,
    "dep_fertilidade_reprodutor": 7.5,
    "heterose_esperada": 4.2,
    "ciclos_sem_concepcao": 1,
}


# ---------------------------------------------------------------------------
# Testes de PredicaoRequest (Pydantic)
# ---------------------------------------------------------------------------


def test_predicao_request_valido():
    req = PredicaoRequest(**MIMOSA_PAYLOAD)
    assert req.especie == "BOVINO"
    assert req.condicao_corporal == 4.0
    assert req.ciclos_sem_concepcao == 1
    assert req.dep_fertilidade_reprodutor == 7.5


def test_predicao_request_cc_acima_limite():
    with pytest.raises(ValidationError) as exc:
        PredicaoRequest(**{**MIMOSA_PAYLOAD, "condicao_corporal": 6.0})
    erros = exc.value.errors()
    assert any("condicao_corporal" in str(e) for e in erros)


def test_predicao_request_cc_abaixo_limite():
    with pytest.raises(ValidationError):
        PredicaoRequest(**{**MIMOSA_PAYLOAD, "condicao_corporal": 0.5})


def test_predicao_request_temperatura_acima_limite():
    with pytest.raises(ValidationError) as exc:
        PredicaoRequest(**{**MIMOSA_PAYLOAD, "temperatura_ambiente_c": 51.0})
    erros = exc.value.errors()
    assert any("temperatura" in str(e) for e in erros)


def test_predicao_request_taxa_prenhez_invalida():
    with pytest.raises(ValidationError):
        PredicaoRequest(**{**MIMOSA_PAYLOAD, "historico_taxa_prenhez": 1.5})


def test_predicao_request_iatf_protocolo_invalido():
    with pytest.raises(ValidationError, match="protocolo"):
        PredicaoRequest(**{**MIMOSA_PAYLOAD,
                          "tipo_inseminacao": "IATF",
                          "protocolo_hormonal": "IA_CONVENCIONAL"})


def test_predicao_request_especie_invalida():
    with pytest.raises(ValidationError):
        PredicaoRequest(**{**MIMOSA_PAYLOAD, "especie": "EQUINO"})


def test_predicao_request_defaults_extras():
    payload_sem_extras = {k: v for k, v in MIMOSA_PAYLOAD.items()
                          if k not in ("ciclos_sem_concepcao",
                                       "dep_fertilidade_reprodutor",
                                       "heterose_esperada")}
    req = PredicaoRequest(**payload_sem_extras)
    assert req.ciclos_sem_concepcao == 0
    assert req.dep_fertilidade_reprodutor is None
    assert req.heterose_esperada is None


# ---------------------------------------------------------------------------
# Testes de PredicaoResponse (Pydantic)
# ---------------------------------------------------------------------------


def test_predicao_response_serializacao():
    import uuid
    resp = PredicaoResponse(
        predicao_id=str(uuid.uuid4()),
        score_prenhez=0.78,
        score_percentual=78,
        classificacao="FAVORAVEL",
        fatores_determinantes=[
            FatorDeterminante(feature="condicao_corporal", valor=4,
                              impacto=0.18, sentido="positivo"),
        ],
        recomendacoes=["Rec 1", "Rec 2", "Rec 3"],
        aviso_clinico="Aviso.",
        modelo_versao="rf_v1.0",
        motor_utilizado="ml",
        processado_em_ms=180,
    )
    d = resp.model_dump()
    assert d["classificacao"] == "FAVORAVEL"
    assert len(d["fatores_determinantes"]) == 1
    assert d["motor_utilizado"] == "ml"


# ---------------------------------------------------------------------------
# Testes de PadroesRequest (Pydantic)
# ---------------------------------------------------------------------------


def test_padroes_request_poucos_registros():
    with pytest.raises(ValidationError, match="20"):
        PadroesRequest(inseminacoes=[{"x": 1}] * 10)


def test_padroes_request_min_maior_que_max():
    from numpy import random as nprng
    ins = [{"x": 1}] * 20
    with pytest.raises(ValidationError, match="min_clusters"):
        PadroesRequest(inseminacoes=ins, min_clusters=5, max_clusters=3)


def test_padroes_request_valido():
    req = PadroesRequest(inseminacoes=[{"x": i} for i in range(20)])
    assert req.min_clusters == 3
    assert req.max_clusters == 5


# ---------------------------------------------------------------------------
# Testes de integração via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from src.serve import app
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "timestamp" in body


def test_predicao_retorna_200(client):
    r = client.post("/predicao", json=MIMOSA_PAYLOAD)
    assert r.status_code == 200
    p = r.json()
    assert 0.0 <= p["score_prenhez"] <= 1.0
    assert p["classificacao"] in ("FAVORAVEL", "MEDIO", "DESFAVORAVEL")
    assert len(p["fatores_determinantes"]) == 5
    assert len(p["recomendacoes"]) == 3
    assert p["aviso_clinico"] != ""
    assert p["motor_utilizado"] in ("ml", "rules")


def test_predicao_cc_invalida_retorna_422(client):
    r = client.post("/predicao", json={**MIMOSA_PAYLOAD, "condicao_corporal": 9.0})
    assert r.status_code == 422


def test_predicao_campo_obrigatorio_ausente_retorna_422(client):
    payload_incompleto = {k: v for k, v in MIMOSA_PAYLOAD.items() if k != "especie"}
    r = client.post("/predicao", json=payload_incompleto)
    assert r.status_code == 422


def test_predicao_fallback_motor_regras(client):
    from src import serve as srv
    estado_original = srv._state["carregado"]
    srv._state["carregado"] = False
    try:
        r = client.post("/predicao", json=MIMOSA_PAYLOAD)
        assert r.status_code == 200
        assert r.json()["motor_utilizado"] == "rules"
    finally:
        srv._state["carregado"] = estado_original


def test_model_info_sem_modelo_retorna_503(client):
    from src import serve as srv
    estado_original = srv._state["carregado"]
    card_original = srv._state["model_card"]
    srv._state["carregado"] = False
    srv._state["model_card"] = None
    try:
        r = client.get("/model-info")
        assert r.status_code == 503
    finally:
        srv._state["carregado"] = estado_original
        srv._state["model_card"] = card_original


def test_model_info_retorna_versao(client):
    r = client.get("/model-info")
    if r.status_code == 200:
        assert "modelo_versao" in r.json()
