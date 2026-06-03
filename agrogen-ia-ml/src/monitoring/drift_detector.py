"""
Detecção de drift estatístico — AgroGen IA (seção 4.4).

Compara a distribuição das últimas 200 predições com as estatísticas
do conjunto de treino para identificar mudanças que possam indicar
degradação do modelo.

  - Kolmogorov-Smirnov (KS) para features contínuas
  - Qui-quadrado para features categóricas

Dispara alerta quando ≥ 3 features apresentam p-valor < 0.05.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import chisquare, ks_2samp

from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

P_VALOR_LIMIAR = 0.05
FEATURES_COM_DRIFT_ALERTA = 3


def comparar_distribuicoes(
    recentes: list[dict],
    training_stats: dict,
) -> dict:
    """
    Compara distribuição das predições recentes com as estatísticas do treino.

    Args:
        recentes: lista de dicts com os campos de features das últimas predições
                  (tipicamente as últimas 200 de tb_predicao_log)
        training_stats: dict com:
            "numerical"   → {feature: [lista de valores do treino]}
            "categorical" → {feature: {categoria: frequência_relativa}}

    Returns:
        Dict com:
            "por_feature"        → resultado de cada feature testada
            "features_com_drift" → contagem de features com p < 0.05
            "alerta"             → True se ≥ 3 features em drift
    """
    if not recentes:
        return {"por_feature": {}, "features_com_drift": 0, "alerta": False}

    resultados: dict[str, dict] = {}

    # ---- Features numéricas (KS test) ----
    train_num = training_stats.get("numerical", {})
    for feat in NUMERICAL_FEATURES:
        vals_recentes = [
            float(r[feat]) for r in recentes
            if feat in r and r[feat] is not None
        ]
        vals_treino = [float(v) for v in train_num.get(feat, []) if v is not None]

        if len(vals_recentes) < 5 or len(vals_treino) < 5:
            continue

        stat, p_val = ks_2samp(vals_recentes, vals_treino)
        resultados[feat] = {
            "drift": p_val < P_VALOR_LIMIAR,
            "p_valor": round(float(p_val), 6),
            "estatistica": round(float(stat), 6),
            "teste": "ks",
            "n_recentes": len(vals_recentes),
            "n_treino": len(vals_treino),
        }

    # ---- Features categóricas (qui-quadrado) ----
    train_cat = training_stats.get("categorical", {})
    for feat in CATEGORICAL_FEATURES:
        vals_recentes = [
            str(r[feat]) for r in recentes
            if feat in r and r[feat] is not None
        ]
        freqs_treino = train_cat.get(feat, {})

        if not vals_recentes or not freqs_treino:
            continue

        cats = list(freqs_treino.keys())
        contagem_recente = Counter(vals_recentes)

        observado = np.array([contagem_recente.get(c, 0) for c in cats], dtype=float)
        esperado_rel = np.array([freqs_treino.get(c, 0.0) for c in cats], dtype=float)

        total_obs = observado.sum()
        total_esp = esperado_rel.sum()
        if total_obs == 0 or total_esp == 0:
            continue

        # Normaliza esperado para a mesma escala do observado
        esperado = esperado_rel * (total_obs / total_esp)

        # Evita zeros no esperado (chisquare falha com frequência esperada = 0)
        if (esperado == 0).any():
            continue

        stat, p_val = chisquare(observado, f_exp=esperado)
        resultados[feat] = {
            "drift": p_val < P_VALOR_LIMIAR,
            "p_valor": round(float(p_val), 6),
            "estatistica": round(float(stat), 6),
            "teste": "chisq",
            "n_recentes": int(total_obs),
        }

    n_drift = sum(1 for r in resultados.values() if r["drift"])

    return {
        "por_feature": resultados,
        "features_com_drift": n_drift,
        "alerta": n_drift >= FEATURES_COM_DRIFT_ALERTA,
    }


def extrair_stats_treino(df_treino) -> dict:
    """
    Extrai training_stats de um DataFrame de treino para uso no drift detector.
    Salve o resultado em model_card.json para consultas futuras.
    """
    import pandas as pd

    stats: dict = {"numerical": {}, "categorical": {}}

    for feat in NUMERICAL_FEATURES:
        if feat in df_treino.columns:
            vals = df_treino[feat].dropna().tolist()
            stats["numerical"][feat] = [float(v) for v in vals]

    for feat in CATEGORICAL_FEATURES:
        if feat in df_treino.columns:
            freq = df_treino[feat].value_counts(normalize=True).to_dict()
            stats["categorical"][feat] = {str(k): float(v) for k, v in freq.items()}

    return stats
