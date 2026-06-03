"""
Testes do motor de regras determinístico.

Cobre todos os grupos de deltas da tabela 3.3 (Matriz-IA.pdf),
o teste de aceitação da Mimosa, e os casos extremos.
"""

import math

import pytest

from src.rules_engine import (
    FATOR_ESCALA,
    PROB_BASE_ESPECIE,
    calcular_deltas,
    calcular_score,
    predict,
)


# ---------------------------------------------------------------------------
# Fixture base neutra (zero deltas esperados além do base)
# ---------------------------------------------------------------------------

BASE_NEUTRA = {
    "especie": "BOVINO",
    "raca_femea": "Angus",           # não adaptada → 0
    "condicao_corporal": 3.5,        # 3.0-4.0 → +10 (não é neutro!)
    "intervalo_pos_parto_dias": 52,  # 45-59 → 0
    "num_partos_anteriores": 1,      # não 0, não 2-4, não ≥7 → 0
    "historico_taxa_prenhez": 0.55,  # 0.40-0.69 → 0
    "dias_desde_ultima_ins": 30,     # ≥ 21 (bovino) → 0
    "dep_fertilidade_animal": 4.0,
    "dep_fertilidade_reprodutor": 4.0,  # soma 8 < 12 → 0
    "dep_acuracia": 0.70,            # ≥ 0.40 → full delta
    "coeficiente_endogamia": 0.01,   # < 0.0625 → 0
    "heterose_esperada": 1.0,        # < 4% → 0
    "tipo_inseminacao": "IA_CONVENCIONAL",  # → 0
    "protocolo_hormonal": "IA_CONVENCIONAL",
    "temperatura_ambiente_c": 25.0,  # 18-28 → 0
    "estacao": "SECA",
    "ciclos_sem_concepcao": 0,       # → 0
}


def soma_deltas(input_data: dict) -> float:
    return sum(d for _, _, d in calcular_deltas(input_data))


# ---------------------------------------------------------------------------
# Condição corporal
# ---------------------------------------------------------------------------

def test_delta_cc_ideal():
    d = {**BASE_NEUTRA, "condicao_corporal": 3.5}
    deltas = calcular_deltas(d)
    cc_delta = next((v for f, _, v in deltas if f == "condicao_corporal"), None)
    assert cc_delta == 10

def test_delta_cc_intermediario():
    d = {**BASE_NEUTRA, "condicao_corporal": 2.5}
    deltas = calcular_deltas(d)
    cc_delta = next((v for f, _, v in deltas if f == "condicao_corporal"), None)
    assert cc_delta == 3

def test_delta_cc_muito_baixo():
    d = {**BASE_NEUTRA, "condicao_corporal": 2.0}
    deltas = calcular_deltas(d)
    cc_delta = next((v for f, _, v in deltas if f == "condicao_corporal"), None)
    assert cc_delta == -12

def test_delta_cc_muito_alto():
    d = {**BASE_NEUTRA, "condicao_corporal": 5.0}
    deltas = calcular_deltas(d)
    cc_delta = next((v for f, _, v in deltas if f == "condicao_corporal"), None)
    assert cc_delta == -12


# ---------------------------------------------------------------------------
# Intervalo pós-parto
# ---------------------------------------------------------------------------

def test_delta_ipp_adequado():
    d = {**BASE_NEUTRA, "intervalo_pos_parto_dias": 90}
    deltas = calcular_deltas(d)
    ipp_delta = next((v for f, _, v in deltas if f == "intervalo_pos_parto_dias"), None)
    assert ipp_delta == 4

def test_delta_ipp_neutro():
    d = {**BASE_NEUTRA, "intervalo_pos_parto_dias": 52}
    deltas = calcular_deltas(d)
    ipp_delta = next((v for f, _, v in deltas if f == "intervalo_pos_parto_dias"), 0)
    assert ipp_delta == 0

def test_delta_ipp_curto():
    d = {**BASE_NEUTRA, "intervalo_pos_parto_dias": 30}
    deltas = calcular_deltas(d)
    ipp_delta = next((v for f, _, v in deltas if f == "intervalo_pos_parto_dias"), None)
    assert ipp_delta == -15


# ---------------------------------------------------------------------------
# Partos anteriores
# ---------------------------------------------------------------------------

def test_delta_primiparidade():
    d = {**BASE_NEUTRA, "num_partos_anteriores": 0}
    deltas = calcular_deltas(d)
    p_delta = next((v for f, _, v in deltas if f == "num_partos_anteriores"), None)
    assert p_delta == -4

def test_delta_partos_favoravel():
    d = {**BASE_NEUTRA, "num_partos_anteriores": 3}
    deltas = calcular_deltas(d)
    p_delta = next((v for f, _, v in deltas if f == "num_partos_anteriores"), None)
    assert p_delta == 6

def test_delta_muitos_partos():
    d = {**BASE_NEUTRA, "num_partos_anteriores": 8}
    deltas = calcular_deltas(d)
    p_delta = next((v for f, _, v in deltas if f == "num_partos_anteriores"), None)
    assert p_delta == -5


# ---------------------------------------------------------------------------
# Histórico taxa de prenhez
# ---------------------------------------------------------------------------

def test_delta_taxa_alta():
    d = {**BASE_NEUTRA, "historico_taxa_prenhez": 0.80}
    deltas = calcular_deltas(d)
    t_delta = next((v for f, _, v in deltas if f == "historico_taxa_prenhez"), None)
    assert t_delta == 8

def test_delta_taxa_baixa():
    d = {**BASE_NEUTRA, "historico_taxa_prenhez": 0.30}
    deltas = calcular_deltas(d)
    t_delta = next((v for f, _, v in deltas if f == "historico_taxa_prenhez"), None)
    assert t_delta == -7


# ---------------------------------------------------------------------------
# Dias desde última inseminação
# ---------------------------------------------------------------------------

def test_delta_dias_ins_bovino_antes_ciclo():
    d = {**BASE_NEUTRA, "especie": "BOVINO", "dias_desde_ultima_ins": 15}
    deltas = calcular_deltas(d)
    d_delta = next((v for f, _, v in deltas if f == "dias_desde_ultima_ins"), None)
    assert d_delta == -10

def test_delta_dias_ins_ovino_antes_ciclo():
    d = {**BASE_NEUTRA, "especie": "OVINO", "dias_desde_ultima_ins": 12}
    deltas = calcular_deltas(d)
    d_delta = next((v for f, _, v in deltas if f == "dias_desde_ultima_ins"), None)
    assert d_delta == -10

def test_delta_dias_ins_apos_ciclo_neutro():
    d = {**BASE_NEUTRA, "especie": "BOVINO", "dias_desde_ultima_ins": 25}
    deltas = calcular_deltas(d)
    d_delta = next((v for f, _, v in deltas if f == "dias_desde_ultima_ins"), 0)
    assert d_delta == 0


# ---------------------------------------------------------------------------
# Tipo inseminação, temperatura, raça
# ---------------------------------------------------------------------------

def test_delta_iatf():
    d = {**BASE_NEUTRA, "tipo_inseminacao": "IATF"}
    deltas = calcular_deltas(d)
    t_delta = next((v for f, _, v in deltas if f == "tipo_inseminacao"), None)
    assert t_delta == 5

def test_delta_temperatura_moderada():
    d = {**BASE_NEUTRA, "temperatura_ambiente_c": 31.0}
    deltas = calcular_deltas(d)
    t_delta = next((v for f, _, v in deltas if f == "temperatura_ambiente_c"), None)
    assert t_delta == -2

def test_delta_temperatura_estresse():
    d = {**BASE_NEUTRA, "temperatura_ambiente_c": 36.0}
    deltas = calcular_deltas(d)
    t_delta = next((v for f, _, v in deltas if f == "temperatura_ambiente_c"), None)
    assert t_delta == -8

def test_delta_raca_adaptada():
    d = {**BASE_NEUTRA, "raca_femea": "Nelore"}
    deltas = calcular_deltas(d)
    r_delta = next((v for f, _, v in deltas if f == "raca_femea"), None)
    assert r_delta == 4


# ---------------------------------------------------------------------------
# Endogamia, heterose, DEP
# ---------------------------------------------------------------------------

def test_delta_endogamia_alta():
    d = {**BASE_NEUTRA, "coeficiente_endogamia": 0.10}
    deltas = calcular_deltas(d)
    e_delta = next((v for f, _, v in deltas if f == "coeficiente_endogamia"), None)
    assert e_delta == -10

def test_delta_heterose_positiva():
    d = {**BASE_NEUTRA, "heterose_esperada": 5.0}
    deltas = calcular_deltas(d)
    h_delta = next((v for f, _, v in deltas if f == "heterose_esperada"), None)
    assert h_delta == 4

def test_delta_dep_soma_favoravel():
    d = {**BASE_NEUTRA, "dep_fertilidade_animal": 7.0, "dep_fertilidade_reprodutor": 6.0}
    deltas = calcular_deltas(d)
    dep_delta = next((v for f, _, v in deltas if f == "dep_fertilidade_animal"), None)
    assert dep_delta == 6  # 7+6=13 ≥ 12

def test_delta_dep_acuracia_aplica_metade():
    d = {**BASE_NEUTRA,
         "dep_fertilidade_animal": 7.0, "dep_fertilidade_reprodutor": 6.0,
         "dep_acuracia": 0.30}  # < 0.40 → metade
    deltas = calcular_deltas(d)
    dep_delta = next((v for f, _, v in deltas if f == "dep_fertilidade_animal"), None)
    assert dep_delta == 3.0  # metade de 6


# ---------------------------------------------------------------------------
# Ciclos sem concepção (extra)
# ---------------------------------------------------------------------------

def test_delta_ciclos_dois():
    d = {**BASE_NEUTRA, "ciclos_sem_concepcao": 2}
    deltas = calcular_deltas(d)
    c_delta = next((v for f, _, v in deltas if f == "ciclos_sem_concepcao"), None)
    assert c_delta == -3

def test_delta_ciclos_tres_ou_mais():
    d = {**BASE_NEUTRA, "ciclos_sem_concepcao": 4}
    deltas = calcular_deltas(d)
    c_delta = next((v for f, _, v in deltas if f == "ciclos_sem_concepcao"), None)
    assert c_delta == -6


# ---------------------------------------------------------------------------
# Fórmula sigmoide
# ---------------------------------------------------------------------------

def test_calcular_score_base_bovino():
    # Σdeltas = 0 → score ≈ prob_base
    score = calcular_score(0.0, "BOVINO", ruido=0.0)
    assert abs(score - 0.60) < 0.001

def test_calcular_score_base_ovino():
    score = calcular_score(0.0, "OVINO", ruido=0.0)
    assert abs(score - 0.65) < 0.001

def test_calcular_score_nao_satura():
    # Score máximo nunca atinge 1.0
    score_max = calcular_score(200.0, "BOVINO", ruido=0.0)
    assert score_max < 1.0

    # Score mínimo nunca atinge 0.0
    score_min = calcular_score(-200.0, "BOVINO", ruido=0.0)
    assert score_min > 0.0


# ---------------------------------------------------------------------------
# Teste de aceitação — Mimosa (seção 5 do Matriz-IA.pdf)
# ---------------------------------------------------------------------------

MIMOSA = {
    "especie": "BOVINO", "raca_femea": "Nelore",
    "condicao_corporal": 4, "num_partos_anteriores": 3,
    "intervalo_pos_parto_dias": 90, "historico_taxa_prenhez": 0.75,
    "dias_desde_ultima_ins": 35, "dep_fertilidade_animal": 8.2,
    "dep_fertilidade_reprodutor": 7.5, "dep_acuracia": 0.80,
    "coeficiente_endogamia": 0.012, "heterose_esperada": 4.2,
    "tipo_inseminacao": "IATF", "protocolo_hormonal": "P4+EB 7 dias",
    "temperatura_ambiente_c": 29, "estacao": "SECA",
    "ciclos_sem_concepcao": 1,
}

def test_mimosa_soma_deltas():
    deltas = calcular_deltas(MIMOSA)
    soma = sum(d for _, _, d in deltas)
    assert soma == 45, f"Σdeltas esperado=45, obtido={soma}"

def test_mimosa_score_sem_ruido():
    deltas = calcular_deltas(MIMOSA)
    soma = sum(d for _, _, d in deltas)
    score = calcular_score(soma, "BOVINO", ruido=0.0)
    assert 0.75 <= score <= 0.84, f"Score Mimosa={score:.4f} fora de [0.75, 0.84]"

def test_mimosa_predict_com_ruido():
    import random
    random.seed(42)
    scores = [predict(MIMOSA, aplicar_ruido=True)["score_prenhez"] for _ in range(50)]
    assert all(0.75 <= s <= 0.86 for s in scores), \
        f"Scores fora do range: min={min(scores):.3f} max={max(scores):.3f}"

def test_mimosa_classificacao_favoravel():
    resultado = predict(MIMOSA, aplicar_ruido=False)
    assert resultado["classificacao"] == "FAVORAVEL"


# ---------------------------------------------------------------------------
# Casos extremos
# ---------------------------------------------------------------------------

PERFEITO = {
    "especie": "BOVINO", "raca_femea": "Nelore",
    "condicao_corporal": 4.0, "num_partos_anteriores": 3,
    "intervalo_pos_parto_dias": 90, "historico_taxa_prenhez": 0.90,
    "dias_desde_ultima_ins": 30, "dep_fertilidade_animal": 9.0,
    "dep_fertilidade_reprodutor": 9.0, "dep_acuracia": 0.90,
    "coeficiente_endogamia": 0.0, "heterose_esperada": 6.0,
    "tipo_inseminacao": "IATF", "protocolo_hormonal": "P4+EB 7 dias",
    "temperatura_ambiente_c": 24, "estacao": "CHUVOSA",
    "ciclos_sem_concepcao": 0,
}

PESSIMO = {
    "especie": "BOVINO", "raca_femea": "Simmental",
    "condicao_corporal": 2.0, "num_partos_anteriores": 0,
    "intervalo_pos_parto_dias": 25, "historico_taxa_prenhez": 0.20,
    "dias_desde_ultima_ins": 10, "dep_fertilidade_animal": 2.0,
    "dep_fertilidade_reprodutor": 2.0, "dep_acuracia": 0.30,
    "coeficiente_endogamia": 0.15, "heterose_esperada": 0.0,
    "tipo_inseminacao": "IA_CONVENCIONAL", "protocolo_hormonal": "IA_CONVENCIONAL",
    "temperatura_ambiente_c": 38, "estacao": "SECA",
    "ciclos_sem_concepcao": 4,
}

def test_animal_perfeito_score_abaixo_095():
    score = predict(PERFEITO, aplicar_ruido=False)["score_prenhez"]
    assert score < 0.95, f"Sigmoide não deve saturar: score={score}"

def test_animal_pessimo_score_acima_005():
    score = predict(PESSIMO, aplicar_ruido=False)["score_prenhez"]
    assert score > 0.05, f"Sigmoide não deve saturar em zero: score={score}"

def test_animal_pessimo_classificacao_desfavoravel():
    resultado = predict(PESSIMO, aplicar_ruido=False)
    assert resultado["classificacao"] == "DESFAVORAVEL"


# ---------------------------------------------------------------------------
# Estrutura do retorno
# ---------------------------------------------------------------------------

def test_predict_retorna_campos_obrigatorios():
    resultado = predict(MIMOSA, aplicar_ruido=False)
    assert "score_prenhez" in resultado
    assert "score_percentual" in resultado
    assert "classificacao" in resultado
    assert "fatores_determinantes" in resultado
    assert "motor_utilizado" in resultado
    assert resultado["motor_utilizado"] == "rules"

def test_predict_fatores_ordenados_por_impacto():
    resultado = predict(MIMOSA, top_k=5, aplicar_ruido=False)
    fatores = resultado["fatores_determinantes"]
    impactos = [abs(f["impacto"]) for f in fatores]
    assert impactos == sorted(impactos, reverse=True), \
        "Fatores devem estar ordenados por |impacto| decrescente"
