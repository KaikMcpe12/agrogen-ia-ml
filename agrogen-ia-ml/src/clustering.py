"""
Análise de padrões de fertilidade do rebanho via K-Means.

O microsserviço recebe o dataset do backend Java e retorna clusters
com descrições textuais geradas por templates — nunca LLM.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.schemas import ClusterInsight, PadroesRequest, PadroesResponse

logger = logging.getLogger(__name__)

FEATURES_CLUSTERING = [
    "condicao_corporal",
    "intervalo_pos_parto_dias",
    "num_partos_anteriores",
    "historico_taxa_prenhez",
    "temperatura_ambiente_c",
]

TARGET_COL = "resultado"  # "PRENHA" ou "VAZIA"


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------


def analisar_padroes(req: PadroesRequest) -> PadroesResponse:
    df = pd.DataFrame(req.inseminacoes)

    # Garante colunas mínimas
    _validar_colunas(df)

    X = df[FEATURES_CLUSTERING].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = _escolher_k(X_scaled, req.min_clusters, req.max_clusters)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    clusters = [_descrever_cluster(df, cid) for cid in range(k) if (df["cluster"] == cid).any()]
    insights = _gerar_insights(clusters)

    return PadroesResponse(
        total_inseminacoes_analisadas=len(df),
        clusters=clusters,
        insights_principais=insights,
        metodologia={
            "algoritmo": "K-Means",
            "n_clusters": k,
            "features": FEATURES_CLUSTERING,
            "normalizacao": "StandardScaler",
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validar_colunas(df: pd.DataFrame) -> None:
    faltando = [c for c in FEATURES_CLUSTERING if c not in df.columns]
    if faltando:
        raise ValueError(
            f"Colunas obrigatórias ausentes no dataset: {faltando}"
        )
    if TARGET_COL not in df.columns:
        # Tenta variantes
        for alt in ["prenha", "diagnostico", "resultado_diagnostico"]:
            if alt in df.columns:
                df[TARGET_COL] = df[alt]
                return
        raise ValueError(
            f"Coluna de resultado ('{TARGET_COL}') não encontrada no dataset."
        )


def _escolher_k(X_scaled: np.ndarray, k_min: int, k_max: int) -> int:
    """Método do cotovelo simplificado."""
    if k_min == k_max:
        return k_min
    inercias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    quedas = [inercias[i] - inercias[i + 1] for i in range(len(inercias) - 1)]
    if not quedas:
        return k_min
    return ks[quedas.index(max(quedas))]


def _descrever_cluster(df: pd.DataFrame, cid: int) -> ClusterInsight:
    sub = df[df["cluster"] == cid]
    perfil = {feat: round(float(sub[feat].mean()), 2) for feat in FEATURES_CLUSTERING}

    if TARGET_COL in sub.columns:
        prenhas = sub[TARGET_COL].isin(["PRENHA", 1, True, "1"]).sum()
        taxa = float(prenhas) / len(sub)
    else:
        taxa = 0.0

    descricao = _template_cluster(perfil, taxa, len(sub))

    return ClusterInsight(
        cluster_id=cid,
        tamanho=len(sub),
        perfil=perfil,
        taxa_prenhez=round(taxa, 3),
        descricao_textual=descricao,
    )


def _template_cluster(perfil: dict, taxa: float, tamanho: int) -> str:
    cc = perfil.get("condicao_corporal", 3.0)
    ipp = perfil.get("intervalo_pos_parto_dias", 60.0)
    temp = perfil.get("temperatura_ambiente_c", 27.0)

    cc_label = "alta" if cc >= 3.5 else ("baixa" if cc < 2.5 else "média")
    ipp_label = "adequado (≥60d)" if ipp >= 60 else "curto (<60d)"
    temp_label = "elevada" if temp >= 34 else ("moderada" if temp >= 29 else "amena")

    return (
        f"Grupo de {tamanho} animais com condição corporal {cc_label} (média {cc:.1f}), "
        f"intervalo pós-parto {ipp_label} (média {ipp:.0f} dias) e temperatura {temp_label} "
        f"({temp:.1f}°C). Taxa de prenhez observada: {taxa*100:.0f}%."
    )


def _gerar_insights(clusters: list[ClusterInsight]) -> list[str]:
    """Identifica padrões entre clusters e gera até 5 frases-chave."""
    insights: list[str] = []
    if len(clusters) < 2:
        return insights

    melhor = max(clusters, key=lambda c: c.taxa_prenhez)
    pior = min(clusters, key=lambda c: c.taxa_prenhez)
    diff = melhor.taxa_prenhez - pior.taxa_prenhez

    if diff > 0.15:
        insights.append(
            f"O grupo com '{melhor.descricao_textual.split('.')[0].lower()}' "
            f"apresenta taxa de prenhez {diff*100:.0f}pp superior ao grupo de pior desempenho."
        )

    # Padrão CC × taxa
    for c in clusters:
        if 3.0 <= c.perfil.get("condicao_corporal", 0) <= 4.0 and c.taxa_prenhez > 0.65:
            insights.append(
                f"Animais com condição corporal entre 3 e 4 atingem "
                f"{c.taxa_prenhez*100:.0f}% de prenhez neste rebanho."
            )
            break

    # Padrão IPP × taxa
    for c in clusters:
        if c.perfil.get("intervalo_pos_parto_dias", 0) >= 60 and c.taxa_prenhez > 0.60:
            insights.append(
                f"Respeitar o intervalo pós-parto mínimo de 60 dias está associado "
                f"a {c.taxa_prenhez*100:.0f}% de sucesso reprodutivo."
            )
            break

    # Padrão temperatura
    for c in clusters:
        if c.perfil.get("temperatura_ambiente_c", 25) >= 34 and c.taxa_prenhez < 0.50:
            insights.append(
                "Inseminações realizadas em condição de estresse calórico (≥34°C) "
                "apresentam taxa significativamente inferior — considere ajuste de calendário."
            )
            break

    return insights[:5]
