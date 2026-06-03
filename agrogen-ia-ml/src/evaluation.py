"""
Métricas de avaliação do modelo de predição de prenhez.

Calcula as 5 métricas exigidas pela seção 4.1 e rejeita modelos que
não atingem os critérios mínimos de promoção a produção.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Critérios mínimos para promoção a produção (seção 4.1)
# ---------------------------------------------------------------------------

CRITERIOS_MINIMOS: dict[str, float] = {
    "acuracia": 0.75,
    "auc_roc": 0.80,
    "f1_prenha": 0.78,
    "precisao_prenha": 0.75,
    "recall_prenha": 0.75,
}

# Queda máxima permitida em qualquer espécie vs acurácia geral (anti-viés regional)
MAX_QUEDA_ESPECIE: float = 0.05


class ModelRejectedError(Exception):
    """Levantado quando o modelo não atinge os critérios mínimos."""

    def __init__(self, falhas: list[str]) -> None:
        self.falhas = falhas
        super().__init__(
            "Modelo rejeitado — critérios não atingidos:\n"
            + "\n".join(f"  • {f}" for f in falhas)
        )


def avaliar(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    especies_test: np.ndarray | None = None,
) -> dict:
    """
    Avalia o modelo no conjunto de teste com as 5 métricas obrigatórias.

    Args:
        model: estimador sklearn com predict e predict_proba
        X_test: features do conjunto de teste, já pré-processadas
        y_test: labels binárias (1 = PRENHA, 0 = VAZIA)
        especies_test: array de strings com a espécie de cada amostra (para anti-viés)

    Returns:
        Dict com as métricas calculadas.

    Raises:
        ModelRejectedError: se alguma métrica não atinge o critério mínimo.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metricas: dict = {
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "f1_prenha": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "precisao_prenha": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_prenha": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }

    # Acurácia por espécie (anti-viés regional)
    if especies_test is not None:
        acuracia_por_especie: dict[str, float] = {}
        for esp in np.unique(especies_test):
            mask = especies_test == esp
            if mask.sum() == 0:
                continue
            acuracia_por_especie[str(esp)] = float(
                accuracy_score(y_test[mask], y_pred[mask])
            )
        metricas["acuracia_por_especie"] = acuracia_por_especie

    _verificar_criterios(metricas)
    return metricas


def _verificar_criterios(metricas: dict) -> None:
    """Levanta ModelRejectedError se algum critério mínimo não for atingido."""
    falhas: list[str] = []

    for metrica, minimo in CRITERIOS_MINIMOS.items():
        valor = metricas.get(metrica, 0.0)
        if valor < minimo:
            falhas.append(
                f"{metrica} = {valor:.4f} < {minimo:.2f} (mínimo exigido)"
            )

    # Anti-viés: nenhuma espécie pode cair mais de 5pp vs acurácia geral
    acuracia_geral = metricas.get("acuracia", 0.0)
    for esp, acc in metricas.get("acuracia_por_especie", {}).items():
        queda = acuracia_geral - acc
        if queda > MAX_QUEDA_ESPECIE:
            falhas.append(
                f"acuracia_{esp.lower()} = {acc:.4f} "
                f"(queda {queda:.4f} > {MAX_QUEDA_ESPECIE:.2f} vs geral)"
            )

    if falhas:
        raise ModelRejectedError(falhas)


def imprimir_relatorio(metricas: dict) -> None:
    """Imprime relatório formatado de métricas no stdout."""
    print("\n" + "=" * 50)
    print("RELATÓRIO DE AVALIAÇÃO DO MODELO")
    print("=" * 50)
    for chave, minimo in CRITERIOS_MINIMOS.items():
        valor = metricas.get(chave, 0.0)
        status = "✓" if valor >= minimo else "✗"
        print(f"  {status} {chave:<22} {valor:.4f}  (mín: {minimo:.2f})")

    if "acuracia_por_especie" in metricas:
        print("\n  Acurácia por espécie:")
        for esp, acc in sorted(metricas["acuracia_por_especie"].items()):
            print(f"    {esp:<10} {acc:.4f}")
    print("=" * 50)
