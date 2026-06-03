"""
Pipeline de pré-processamento do AgroGen IA.

Responsável por:
  - Validação de range das features de entrada
  - Tratamento de ausências (mediana para numéricas, moda para categóricas)
  - One-hot encoding das features categóricas
  - Normalização MinMaxScaler das features numéricas
  - Manutenção da ordem canônica de features para compatibilidade com o modelo
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FEATURES_ML = [
    "condicao_corporal",
    "historico_taxa_prenhez",
    "intervalo_pos_parto_dias",
    "num_partos_anteriores",
    "dias_desde_ultima_ins",
    "dep_fertilidade_animal",
    "especie",
    "raca_femea",
    "tipo_inseminacao",
    "protocolo_hormonal",
    "temperatura_ambiente_c",
    "estacao",
    "dep_acuracia",
    "coeficiente_endogamia",
]

CATEGORICAL_FEATURES = [
    "especie",
    "raca_femea",
    "tipo_inseminacao",
    "protocolo_hormonal",
    "estacao",
]

NUMERICAL_FEATURES = [
    "condicao_corporal",
    "historico_taxa_prenhez",
    "intervalo_pos_parto_dias",
    "num_partos_anteriores",
    "dias_desde_ultima_ins",
    "dep_fertilidade_animal",
    "temperatura_ambiente_c",
    "dep_acuracia",
    "coeficiente_endogamia",
]

# Regras de validação: (min, max) ou None se sem limite num sentido
VALIDATION_RULES: dict[str, tuple] = {
    "condicao_corporal": (1.0, 5.0),
    "historico_taxa_prenhez": (0.0, 1.0),
    "intervalo_pos_parto_dias": (0, None),
    "num_partos_anteriores": (0, None),
    "dias_desde_ultima_ins": (0, None),
    "dep_fertilidade_animal": (None, None),
    "temperatura_ambiente_c": (None, 50.0),
    "dep_acuracia": (0.0, 1.0),
    "coeficiente_endogamia": (0.0, 1.0),
}


class ValidationError(ValueError):
    """Erro estruturado de validação com campo e valor."""

    def __init__(self, field: str, value, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Validação falhou: {field}={value} — {reason}")

    def to_dict(self) -> dict:
        return {"campo": self.field, "valor": self.value, "motivo": self.reason}


class Preprocessor:
    """
    Encapsula todo o pré-processamento: fit no treino, transform na inferência.

    Uso:
        prep = Preprocessor()
        X_train_t = prep.fit_transform(df_train[FEATURES_ML])
        X_test_t  = prep.transform(df_test[FEATURES_ML])
        prep.save("models/preprocessor_v1.0.pkl")

        prep2 = Preprocessor.load("models/preprocessor_v1.0.pkl")
        X_new = prep2.transform(df_new)
    """

    def __init__(self) -> None:
        self._encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self._scaler = MinMaxScaler()
        self._cat_medians: dict[str, str] = {}   # moda por feature categórica
        self._num_medians: dict[str, float] = {}  # mediana por feature numérica
        self._feature_names_out: list[str] = []   # nomes após encoding
        self._fitted = False

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------

    def validate(self, input_dict: dict) -> None:
        """
        Valida os valores de entrada antes do transform.
        Levanta ValidationError no primeiro campo inválido.
        """
        for field, (lo, hi) in VALIDATION_RULES.items():
            if field not in input_dict:
                continue
            val = input_dict[field]
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if lo is not None and v < lo:
                raise ValidationError(field, val, f"deve ser ≥ {lo}")
            if hi is not None and v > hi:
                raise ValidationError(field, val, f"deve ser ≤ {hi}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Ajusta encoders e scaler sobre o conjunto de treino."""
        X = self._fill_missing(X, fit=True)

        # Fit encoder categórico
        self._encoder.fit(X[CATEGORICAL_FEATURES])

        # Fit scaler numérico
        self._scaler.fit(X[NUMERICAL_FEATURES])

        # Nomes das features após encoding (para shap_utils e model_card)
        cat_names = list(self._encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self._feature_names_out = NUMERICAL_FEATURES + cat_names

        self._fitted = True
        return self

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame | dict) -> np.ndarray:
        """
        Aplica o pré-processamento a um DataFrame ou dict de entrada única.
        Retorna ndarray shape (n_samples, n_features_encoded).
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor não foi ajustado. Chame fit() primeiro.")

        if isinstance(X, dict):
            X = pd.DataFrame([X])

        X = X.copy()

        # Garante que apenas as 14 features ML estão presentes
        for col in FEATURES_ML:
            if col not in X.columns:
                X[col] = np.nan

        X = X[FEATURES_ML]
        X = self._fill_missing(X, fit=False)

        num_arr = self._scaler.transform(X[NUMERICAL_FEATURES].astype(np.float32))
        cat_arr = self._encoder.transform(X[CATEGORICAL_FEATURES])

        return np.hstack([num_arr, cat_arr]).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Arquivo não contém um Preprocessor: {path}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_missing(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()

        # Numéricos: mediana
        for col in NUMERICAL_FEATURES:
            if col not in X.columns:
                continue
            if fit:
                self._num_medians[col] = float(X[col].median())
            X[col] = X[col].fillna(self._num_medians.get(col, 0.0))
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self._num_medians.get(col, 0.0))

        # Categóricos: moda
        for col in CATEGORICAL_FEATURES:
            if col not in X.columns:
                continue
            if fit:
                mode_val = X[col].mode()
                self._cat_medians[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "BOVINO"
            X[col] = X[col].fillna(self._cat_medians.get(col, "BOVINO"))
            X[col] = X[col].astype(str)

        return X

    @property
    def feature_names_out(self) -> list[str]:
        """Nomes das features após encoding, na ordem que o modelo recebe."""
        return list(self._feature_names_out)

    @property
    def onehot_categories(self) -> dict[str, list[str]]:
        """Categorias aprendidas pelo OneHotEncoder, por feature."""
        if not self._fitted:
            return {}
        return {
            col: list(cats)
            for col, cats in zip(CATEGORICAL_FEATURES, self._encoder.categories_)
        }
