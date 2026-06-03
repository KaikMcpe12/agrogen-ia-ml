"""
Pipeline de treinamento do modelo Random Forest — AgroGen IA.

Executa as etapas 1–5 do pipeline de cold start (seção 4.1):
  1. Carrega cold_start_v1.csv
  2. Pré-processamento (fit + transform)
  3. Split 80/20 estratificado
  4. GridSearchCV com 5-fold cross-validation
  5. Avaliação, serialização e geração do model_card.json

Uso:
    python src/training.py
    python src/training.py --data data/cold_start_v1.csv --version rf_v1.1
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.evaluation import ModelRejectedError, avaliar, imprimir_relatorio
from src.preprocessing import FEATURES_ML, Preprocessor

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE_DIR / "data" / "cold_start_v1.csv"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_VERSION = "rf_v1.0"

PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

RF_FIXED_PARAMS = {
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def treinar(
    data_path: Path = DEFAULT_DATA,
    models_dir: Path = DEFAULT_MODELS_DIR,
    version: str = DEFAULT_VERSION,
) -> dict:
    """
    Executa o pipeline completo de treinamento.

    Returns:
        Dict com as métricas obtidas no conjunto de teste.

    Raises:
        ModelRejectedError: se o modelo não atingir os critérios mínimos.
        FileNotFoundError: se o dataset não existir.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {data_path}\n"
            "Execute: python data/synthetic_generator.py"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    # ------------------------------------------------------------------
    # 1. Carrega dataset
    # ------------------------------------------------------------------
    print(f"[1/5] Carregando dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      {len(df)} registros, {df['prenha'].mean():.3f} taxa de prenhez")

    sha256 = hashlib.sha256(open(data_path, "rb").read()).hexdigest()

    X = df[FEATURES_ML]
    y = df["prenha"].values
    especies = df["especie"].values

    # ------------------------------------------------------------------
    # 2. Split estratificado 80/20
    # ------------------------------------------------------------------
    print("[2/5] Dividindo treino/teste (80/20 estratificado)")
    X_train, X_test, y_train, y_test, esp_train, esp_test = train_test_split(
        X, y, especies,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"      Treino: {len(X_train)} | Teste: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Pré-processamento
    # ------------------------------------------------------------------
    print("[3/5] Ajustando pré-processador e transformando features")
    prep = Preprocessor()
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)
    print(f"      Shape pós-encoding: {X_train_t.shape[1]} features")

    # ------------------------------------------------------------------
    # 4. GridSearchCV com validação cruzada 5-fold
    # ------------------------------------------------------------------
    print("[4/5] Executando GridSearchCV (5-fold CV) — isso leva ~4 min")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(**RF_FIXED_PARAMS)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train_t, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    model = grid_search.best_estimator_

    print(f"      Melhores hiperparâmetros: {best_params}")
    print(f"      Melhor AUC-ROC (CV): {best_cv_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Avaliação no conjunto de teste
    # ------------------------------------------------------------------
    print("[5/5] Avaliando no conjunto de teste")
    try:
        metricas = avaliar(model, X_test_t, y_test, especies_test=esp_test)
    except ModelRejectedError as exc:
        imprimir_relatorio(exc.falhas and {})  # tipo: silencia, já imprime
        raise

    imprimir_relatorio(metricas)

    # ------------------------------------------------------------------
    # Serialização
    # ------------------------------------------------------------------
    duracao = round(time.time() - inicio)

    model_path = models_dir / f"{version}.pkl"
    prep_path = models_dir / f"{version}_preprocessor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(prep, prep_path)
    print(f"\nModelo salvo em: {model_path}")
    print(f"Preprocessor salvo em: {prep_path}")

    # ------------------------------------------------------------------
    # model_card.json (11 campos — seção 4.2)
    # ------------------------------------------------------------------
    # Feature importance do modelo vencedor
    feat_imp_raw = model.feature_importances_
    feat_names = prep.feature_names_out
    feature_importance = dict(
        sorted(
            zip(feat_names, feat_imp_raw.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    dist_classes = {
        "PRENHA": round(float(y_train.mean()), 4),
        "VAZIA": round(float(1 - y_train.mean()), 4),
    }

    model_card = {
        "modelo_versao": version,
        "tipo": "RandomForestClassifier",
        "framework": f"scikit-learn {_sklearn_version()}",
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "treinado_por": os.environ.get("TRAINER_EMAIL", "equipe@agrogenia.com"),
        "duracao_treinamento_segundos": duracao,
        "dataset": {
            "arquivo": str(data_path.name),
            "hash_sha256": sha256,
            "n_registros": len(df),
            "split": {"treino": len(X_train), "teste": len(X_test)},
            "distribuicao_classes": dist_classes,
        },
        "hiperparametros": {
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "min_samples_split": best_params["min_samples_split"],
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "metricas": {
            "acuracia": round(metricas["acuracia"], 4),
            "auc_roc": round(metricas["auc_roc"], 4),
            "f1_prenha": round(metricas["f1_prenha"], 4),
            "precisao_prenha": round(metricas["precisao_prenha"], 4),
            "recall_prenha": round(metricas["recall_prenha"], 4),
            "acuracia_por_especie": {
                k: round(v, 4)
                for k, v in metricas.get("acuracia_por_especie", {}).items()
            },
        },
        "features_ativas": FEATURES_ML,
        "feature_importance": {k: round(v, 6) for k, v in feature_importance.items()},
        "feature_order": feat_names,
        "preprocessor_params": {
            "scaler_type": "MinMaxScaler",
            "onehot_categories": prep.onehot_categories,
            "numerical_features": prep.feature_names_out[:9],
        },
    }

    card_path = models_dir / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2, ensure_ascii=False))
    print(f"model_card.json salvo em: {card_path}")

    return metricas


def _sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo Random Forest do AgroGen IA")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Caminho do CSV de treino")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Diretório de saída")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION, help="Versão do modelo (ex: rf_v1.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        treinar(data_path=args.data, models_dir=args.models_dir, version=args.version)
        print("\nTreinamento concluído com sucesso.")
    except ModelRejectedError as e:
        print(f"\n{e}")
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        raise SystemExit(1)
