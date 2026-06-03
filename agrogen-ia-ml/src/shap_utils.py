"""
Extração de top-5 fatores determinantes via SHAP (TreeExplainer).

Fallback para feature_importances_ global se shap não estiver instalado.
"""

from __future__ import annotations

import logging

import numpy as np

from src.preprocessing import CATEGORICAL_FEATURES
from src.schemas import FatorDeterminante

logger = logging.getLogger(__name__)

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False
    logger.warning("shap não instalado — usando feature_importances_ global como fallback")


# ---------------------------------------------------------------------------
# Criação do explainer (chamada uma vez no startup)
# ---------------------------------------------------------------------------


def criar_explainer(model):
    """Instancia e retorna um TreeExplainer. Retorna None se shap indisponível."""
    if not SHAP_AVAILABLE:
        return None
    return _shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Extração dos fatores
# ---------------------------------------------------------------------------


def extrair_fatores(
    explainer,
    X_transformed: np.ndarray,
    feature_names_out: list[str],
    valores_originais: dict,
    model=None,
    top_k: int = 5,
) -> list[FatorDeterminante]:
    """
    Retorna top-k fatores que mais contribuíram para esta predição.

    Args:
        explainer: TreeExplainer (ou None para fallback)
        X_transformed: array (1, n_features) pós-encoding
        feature_names_out: nomes das colunas após one-hot, em ordem
        valores_originais: dict com valores brutos (antes do encoding)
        model: estimador sklearn (usado no fallback)
        top_k: número de fatores a retornar

    Returns:
        Lista de FatorDeterminante ordenada por |impacto| decrescente.
    """
    if explainer is not None and SHAP_AVAILABLE:
        return _fatores_via_shap(explainer, X_transformed, feature_names_out, valores_originais, top_k)
    return _fatores_via_importancia_global(model, feature_names_out, valores_originais, top_k)


def _fatores_via_shap(
    explainer,
    X_transformed: np.ndarray,
    feature_names_out: list[str],
    valores_originais: dict,
    top_k: int,
) -> list[FatorDeterminante]:
    shap_vals = explainer.shap_values(X_transformed, check_additivity=False)

    # RandomForestClassifier retorna lista [neg_class, pos_class] ou array único
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]          # classe PRENHA, primeira amostra
    elif shap_vals.ndim == 3:
        vals = shap_vals[0, :, 1]       # (n_samples, n_features, n_classes)
    else:
        vals = shap_vals[0]             # formato moderno (única saída)

    agregado = _agregar_one_hot(vals.tolist(), feature_names_out)
    return _top_k_fatores(agregado, valores_originais, top_k)


def _fatores_via_importancia_global(
    model,
    feature_names_out: list[str],
    valores_originais: dict,
    top_k: int,
) -> list[FatorDeterminante]:
    """Fallback: usa importância global do RF (não por predição)."""
    if model is None:
        return []
    importancias = model.feature_importances_.tolist()
    agregado = _agregar_one_hot(importancias, feature_names_out)
    return _top_k_fatores(agregado, valores_originais, top_k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agregar_one_hot(
    valores: list[float],
    feature_names_out: list[str],
) -> dict[str, float]:
    """
    Agrega contribuições de colunas one-hot de volta ao nome original.

    Ex: especie_BOVINO=0.05, especie_OVINO=-0.01 → especie=0.05+(-0.01)=0.04
    """
    agregado: dict[str, float] = {}
    for name, val in zip(feature_names_out, valores):
        original = name
        for cat in CATEGORICAL_FEATURES:
            if name.startswith(f"{cat}_"):
                original = cat
                break
        agregado[original] = agregado.get(original, 0.0) + float(val)
    return agregado


def _top_k_fatores(
    agregado: dict[str, float],
    valores_originais: dict,
    top_k: int,
) -> list[FatorDeterminante]:
    ordenados = sorted(agregado.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [
        FatorDeterminante(
            feature=feat,
            valor=valores_originais.get(feat),
            impacto=round(imp, 4),
            sentido="positivo" if imp >= 0 else "negativo",
        )
        for feat, imp in ordenados
    ]
