"""
Motor de Regras Determinístico — AgroGen IA.

Implementa o fallback de predição de prenhez via fórmula sigmoide:

    score = sigmoid(Σdeltas / FATOR_ESCALA + logit(prob_base_especie))

Todos os 26 deltas da tabela 3.3 (Matriz-IA.pdf) estão parametrizados
em DELTA_RULES, mais 2 deltas extras para ciclos_sem_concepcao.

Referência: EMBRAPA GENECOC / Hafez 2004.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constantes parametrizáveis
# ---------------------------------------------------------------------------

PROB_BASE_ESPECIE: dict[str, float] = {
    "BOVINO": 0.60,
    "OVINO": 0.65,
    "CAPRINO": 0.65,
}

# Parâmetro de escala: calibrado para que Σdeltas=45 (Mimosa) mapeie para ~0.82.
# Aumentar → curva mais plana (menos sensível). Diminuir → mais steep.
FATOR_ESCALA: float = 40.0

# Raças com vantagem adaptativa ao semiárido nordestino
RACAS_ADAPTADAS: frozenset[str] = frozenset(
    {"Nelore", "Santa Inês", "Anglo-nubiano", "Moxotó"}
)

# Ciclo estral mínimo por espécie (dias); usado no delta de dias_desde_ultima_ins
CICLO_MINIMO: dict[str, int] = {"BOVINO": 21, "OVINO": 17, "CAPRINO": 17}

AVISO_CLINICO: str = (
    "Este score é uma estimativa probabilística baseada em dados históricos "
    "e não substitui o julgamento clínico veterinário."
)

# ---------------------------------------------------------------------------
# Estrutura de regra
# ---------------------------------------------------------------------------


@dataclass
class DeltaRule:
    """
    Regra individual de delta.

    feature   : nome da feature (para exibição ao usuário)
    label     : descrição curta da condição
    condition : função que recebe o input_dict e retorna True se a regra se aplica
    delta     : pontos a adicionar ao score (pode ser negativo)
    """

    feature: str
    label: str
    condition: Callable[[dict], bool]
    delta: float


# ---------------------------------------------------------------------------
# Tabela de deltas (seção 3.3 da Matriz-IA.pdf) + extras ciclos_sem_concepcao
# Ordem: uma regra por condição; condições mutuamente exclusivas dentro do grupo.
# ---------------------------------------------------------------------------

DELTA_RULES: list[DeltaRule] = [
    # --- Condição corporal ---
    DeltaRule(
        "condicao_corporal", "CC 3.0–4.0 (ideal)",
        lambda d: 3.0 <= d.get("condicao_corporal", 3.0) <= 4.0,
        +10,
    ),
    DeltaRule(
        "condicao_corporal", "CC 2.5 ou 4.5",
        lambda d: d.get("condicao_corporal", 3.0) in {2.5, 4.5},
        +3,
    ),
    DeltaRule(
        "condicao_corporal", "CC < 2.5 ou > 4.5",
        lambda d: d.get("condicao_corporal", 3.0) < 2.5 or d.get("condicao_corporal", 3.0) > 4.5,
        -12,
    ),

    # --- Intervalo pós-parto ---
    DeltaRule(
        "intervalo_pos_parto_dias", "IPP ≥ 60 dias",
        lambda d: d.get("intervalo_pos_parto_dias", 60) >= 60,
        +4,
    ),
    DeltaRule(
        "intervalo_pos_parto_dias", "IPP 45–59 dias (neutro)",
        lambda d: 45 <= d.get("intervalo_pos_parto_dias", 60) <= 59,
        0,
    ),
    DeltaRule(
        "intervalo_pos_parto_dias", "IPP < 45 dias",
        lambda d: d.get("intervalo_pos_parto_dias", 60) < 45,
        -15,
    ),

    # --- Número de partos anteriores ---
    DeltaRule(
        "num_partos_anteriores", "Primípara (0 partos)",
        lambda d: d.get("num_partos_anteriores", 1) == 0,
        -4,
    ),
    DeltaRule(
        "num_partos_anteriores", "2 a 4 partos",
        lambda d: 2 <= d.get("num_partos_anteriores", 1) <= 4,
        +6,
    ),
    DeltaRule(
        "num_partos_anteriores", "≥ 7 partos",
        lambda d: d.get("num_partos_anteriores", 1) >= 7,
        -5,
    ),

    # --- Histórico de taxa de prenhez ---
    DeltaRule(
        "historico_taxa_prenhez", "Taxa histórica ≥ 0.70",
        lambda d: d.get("historico_taxa_prenhez", 0.5) >= 0.70,
        +8,
    ),
    DeltaRule(
        "historico_taxa_prenhez", "Taxa histórica 0.40–0.69 (neutro)",
        lambda d: 0.40 <= d.get("historico_taxa_prenhez", 0.5) < 0.70,
        0,
    ),
    DeltaRule(
        "historico_taxa_prenhez", "Taxa histórica < 0.40",
        lambda d: d.get("historico_taxa_prenhez", 0.5) < 0.40,
        -7,
    ),

    # --- Dias desde última inseminação (bovino) ---
    DeltaRule(
        "dias_desde_ultima_ins", "Dias desde última IA < 21 (Bov) — ciclo incompleto",
        lambda d: d.get("especie", "BOVINO") == "BOVINO"
                  and d.get("dias_desde_ultima_ins", 30) < 21,
        -10,
    ),

    # --- Dias desde última inseminação (ovino/caprino) ---
    DeltaRule(
        "dias_desde_ultima_ins", "Dias desde última IA < 17 (O/C) — ciclo incompleto",
        lambda d: d.get("especie", "BOVINO") in {"OVINO", "CAPRINO"}
                  and d.get("dias_desde_ultima_ins", 30) < 17,
        -10,
    ),

    # --- Tipo de inseminação ---
    DeltaRule(
        "tipo_inseminacao", "IATF com protocolo válido",
        lambda d: d.get("tipo_inseminacao", "") == "IATF",
        +5,
    ),
    DeltaRule(
        "tipo_inseminacao", "IA Convencional (neutro)",
        lambda d: d.get("tipo_inseminacao", "") == "IA_CONVENCIONAL",
        0,
    ),

    # --- Temperatura ambiente ---
    DeltaRule(
        "temperatura_ambiente_c", "Temperatura 18–28°C (neutro)",
        lambda d: 18 <= d.get("temperatura_ambiente_c", 25) <= 28,
        0,
    ),
    DeltaRule(
        "temperatura_ambiente_c", "Temperatura 29–33°C",
        lambda d: 29 <= d.get("temperatura_ambiente_c", 25) <= 33,
        -2,
    ),
    DeltaRule(
        "temperatura_ambiente_c", "Temperatura ≥ 34°C (estresse calórico)",
        lambda d: d.get("temperatura_ambiente_c", 25) >= 34,
        -8,
    ),

    # --- Raça da fêmea ---
    DeltaRule(
        "raca_femea", "Raça adaptada ao semiárido",
        lambda d: d.get("raca_femea", "") in RACAS_ADAPTADAS,
        +4,
    ),
    DeltaRule(
        "raca_femea", "Demais raças (neutro)",
        lambda d: d.get("raca_femea", "") not in RACAS_ADAPTADAS,
        0,
    ),

    # --- Heterose esperada ---
    DeltaRule(
        "heterose_esperada", "Heterose ≥ 4% (cruzamento entre raças distantes)",
        lambda d: d.get("heterose_esperada", 0.0) is not None
                  and d.get("heterose_esperada", 0.0) >= 4.0,
        +4,
    ),

    # --- Coeficiente de endogamia ---
    DeltaRule(
        "coeficiente_endogamia", "Endogamia > 0.0625 (consanguinidade elevada)",
        lambda d: d.get("coeficiente_endogamia", 0.0) > 0.0625,
        -10,
    ),

    # --- DEP fertilidade somada (animal + reprodutor) ---
    # dep_acuracia < 0.40 → delta é aplicado pela metade (ver após o loop)
    DeltaRule(
        "dep_fertilidade_animal", "DEP fertilidade somada ≥ 12",
        lambda d: (d.get("dep_fertilidade_animal", 0.0) or 0.0)
                  + (d.get("dep_fertilidade_reprodutor", 0.0) or 0.0) >= 12,
        +6,
    ),

    # --- Ciclos sem concepção (extra — não entra no ML) ---
    DeltaRule(
        "ciclos_sem_concepcao", "≥ 3 ciclos sem concepção (subfertilidade)",
        lambda d: d.get("ciclos_sem_concepcao", 0) >= 3,
        -6,
    ),
    DeltaRule(
        "ciclos_sem_concepcao", "2 ciclos sem concepção",
        lambda d: d.get("ciclos_sem_concepcao", 0) == 2,
        -3,
    ),
]

# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _classificar(score: float) -> str:
    if score >= 0.70:
        return "FAVORAVEL"
    elif score >= 0.50:
        return "MEDIO"
    return "DESFAVORAVEL"


# ---------------------------------------------------------------------------
# Cálculo de deltas
# ---------------------------------------------------------------------------


def calcular_deltas(input_data: dict[str, Any]) -> list[tuple[str, str, float]]:
    """
    Avalia todas as regras sobre input_data.

    Retorna lista de (feature, label, delta_efetivo) para todas as regras
    que se aplicaram com delta != 0.

    O delta do DEP é reduzido à metade se dep_acuracia < 0.40.
    """
    resultados: list[tuple[str, str, float]] = []
    dep_acuracia = float(input_data.get("dep_acuracia", 1.0) or 1.0)

    for rule in DELTA_RULES:
        try:
            aplica = rule.condition(input_data)
        except (TypeError, KeyError):
            aplica = False

        if not aplica:
            continue

        delta_efetivo = rule.delta
        # Regra especial: DEP fertilidade tem delta reduzido se acurácia baixa
        if rule.feature == "dep_fertilidade_animal" and dep_acuracia < 0.40:
            delta_efetivo = rule.delta / 2.0

        resultados.append((rule.feature, rule.label, delta_efetivo))

    return resultados


def calcular_score(soma_deltas: float, especie: str, ruido: float = 0.0) -> float:
    """
    Converte a soma de deltas em score [0, 1] via sigmoide.

    Args:
        soma_deltas: soma algébrica de todos os deltas aplicados
        especie: BOVINO | OVINO | CAPRINO
        ruido: variação aleatória adicional (±0.02 em produção)

    Returns:
        Score no intervalo [0.01, 0.99].
    """
    prob_base = PROB_BASE_ESPECIE.get(especie, 0.60)
    offset = _logit(prob_base)
    score = _sigmoid(soma_deltas / FATOR_ESCALA + offset) + ruido
    return max(0.01, min(0.99, score))


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------


def predict(
    input_data: dict[str, Any],
    top_k: int = 5,
    aplicar_ruido: bool = True,
) -> dict[str, Any]:
    """
    Executa a predição pelo motor de regras.

    Args:
        input_data: dict com as features (aceita os 17 campos do PredicaoRequest)
        top_k: número de fatores determinantes a retornar
        aplicar_ruido: se True, adiciona ruído ±0.02 ao score final

    Returns:
        Dict compatível com PredicaoResponse.
    """
    especie = str(input_data.get("especie", "BOVINO")).upper()

    deltas_aplicados = calcular_deltas(input_data)
    soma = sum(d for _, _, d in deltas_aplicados)

    ruido = random.uniform(-0.02, 0.02) if aplicar_ruido else 0.0
    score = calcular_score(soma, especie, ruido)

    # Top-K fatores: ordenados por |delta| decrescente, apenas não-nulos
    fatores_nao_nulos = [(f, lbl, d) for f, lbl, d in deltas_aplicados if d != 0]
    top_fatores = sorted(fatores_nao_nulos, key=lambda x: abs(x[2]), reverse=True)[:top_k]

    fatores_determinantes = [
        {
            "feature": feat,
            "valor": input_data.get(feat),
            "impacto": round(delta / FATOR_ESCALA, 4),
            "sentido": "positivo" if delta > 0 else "negativo",
            "label": label,
        }
        for feat, label, delta in top_fatores
    ]

    return {
        "score_prenhez": round(score, 4),
        "score_percentual": round(score * 100),
        "classificacao": _classificar(score),
        "fatores_determinantes": fatores_determinantes,
        "aviso_clinico": AVISO_CLINICO,
        "motor_utilizado": "rules",
    }
