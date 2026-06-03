"""Testes do pipeline de pré-processamento."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES_ML,
    NUMERICAL_FEATURES,
    Preprocessor,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_sample() -> pd.DataFrame:
    """DataFrame sintético mínimo com todos os campos ML."""
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "condicao_corporal":        rng.uniform(1.5, 5.0, n),
        "historico_taxa_prenhez":   rng.uniform(0.0, 1.0, n),
        "intervalo_pos_parto_dias": rng.integers(20, 200, n),
        "num_partos_anteriores":    rng.integers(0, 8, n),
        "dias_desde_ultima_ins":    rng.integers(5, 180, n),
        "dep_fertilidade_animal":   rng.uniform(0.0, 15.0, n),
        "especie":                  rng.choice(["BOVINO", "OVINO", "CAPRINO"], n),
        "raca_femea":               rng.choice(["Nelore", "Angus", "Santa Inês"], n),
        "tipo_inseminacao":         rng.choice(["IATF", "IA_CONVENCIONAL"], n),
        "protocolo_hormonal":       rng.choice(["P4+EB 7 dias", "OvSynch", "IA_CONVENCIONAL"], n),
        "temperatura_ambiente_c":   rng.uniform(18.0, 42.0, n),
        "estacao":                  rng.choice(["SECA", "CHUVOSA"], n),
        "dep_acuracia":             rng.uniform(0.1, 0.99, n),
        "coeficiente_endogamia":    rng.uniform(0.0, 0.25, n),
    })


@pytest.fixture
def fitted_prep(df_sample) -> Preprocessor:
    prep = Preprocessor()
    prep.fit(df_sample)
    return prep


# ---------------------------------------------------------------------------
# Testes de shape e tipos
# ---------------------------------------------------------------------------


def test_fit_transform_retorna_ndarray(df_sample):
    prep = Preprocessor()
    X = prep.fit_transform(df_sample)
    assert isinstance(X, np.ndarray)


def test_fit_transform_shape(df_sample):
    prep = Preprocessor()
    X = prep.fit_transform(df_sample)
    assert X.shape[0] == len(df_sample)
    assert X.shape[1] > len(FEATURES_ML)  # one-hot expande


def test_feature_names_count(fitted_prep, df_sample):
    X = fitted_prep.transform(df_sample)
    assert len(fitted_prep.feature_names_out) == X.shape[1]


def test_numericas_apos_scaler_em_zero_um(df_sample):
    prep = Preprocessor()
    X = prep.fit_transform(df_sample)
    n_num = len(NUMERICAL_FEATURES)
    cols_num = X[:, :n_num]
    assert cols_num.min() >= -1e-6, "MinMaxScaler: valor abaixo de 0"
    assert cols_num.max() <= 1 + 1e-6, "MinMaxScaler: valor acima de 1"


def test_onehot_valores_binarios(df_sample):
    prep = Preprocessor()
    X = prep.fit_transform(df_sample)
    n_num = len(NUMERICAL_FEATURES)
    cols_cat = X[:, n_num:]
    unicos = np.unique(cols_cat)
    assert set(unicos).issubset({0.0, 1.0}), "One-hot deve conter apenas 0 e 1"


# ---------------------------------------------------------------------------
# Testes de tratamento de ausências
# ---------------------------------------------------------------------------


def test_ausencia_numerica_preenchida_com_mediana(df_sample):
    prep = Preprocessor()
    prep.fit(df_sample)

    row = df_sample.iloc[[0]].copy()
    row["condicao_corporal"] = np.nan

    X_com_nan = prep.transform(row)
    row_normal = df_sample.iloc[[0]].copy()
    row_normal["condicao_corporal"] = prep._num_medians["condicao_corporal"]
    X_normal = prep.transform(row_normal)

    assert np.allclose(X_com_nan, X_normal, atol=1e-5)


def test_ausencia_categorica_preenchida_com_moda(df_sample):
    prep = Preprocessor()
    prep.fit(df_sample)

    row = df_sample.iloc[[0]].copy()
    row["especie"] = None

    # Não deve lançar exceção
    X = prep.transform(row)
    assert X.shape == (1, len(prep.feature_names_out))


def test_transform_dict_entrada_unica(fitted_prep):
    entrada = {
        "condicao_corporal": 3.5, "historico_taxa_prenhez": 0.65,
        "intervalo_pos_parto_dias": 75, "num_partos_anteriores": 2,
        "dias_desde_ultima_ins": 30, "dep_fertilidade_animal": 7.0,
        "especie": "BOVINO", "raca_femea": "Nelore",
        "tipo_inseminacao": "IATF", "protocolo_hormonal": "P4+EB 7 dias",
        "temperatura_ambiente_c": 27.0, "estacao": "SECA",
        "dep_acuracia": 0.75, "coeficiente_endogamia": 0.01,
    }
    X = fitted_prep.transform(entrada)
    assert X.shape[0] == 1


# ---------------------------------------------------------------------------
# Testes de validação de range
# ---------------------------------------------------------------------------


def test_validate_cc_acima_do_limite(fitted_prep):
    with pytest.raises(ValidationError) as exc_info:
        fitted_prep.validate({"condicao_corporal": 5.5})
    assert exc_info.value.field == "condicao_corporal"


def test_validate_cc_abaixo_do_limite(fitted_prep):
    with pytest.raises(ValidationError) as exc_info:
        fitted_prep.validate({"condicao_corporal": 0.5})
    assert exc_info.value.field == "condicao_corporal"


def test_validate_temperatura_acima_do_limite(fitted_prep):
    with pytest.raises(ValidationError) as exc_info:
        fitted_prep.validate({"temperatura_ambiente_c": 55.0})
    assert exc_info.value.field == "temperatura_ambiente_c"


def test_validate_taxa_prenhez_invalida(fitted_prep):
    with pytest.raises(ValidationError):
        fitted_prep.validate({"historico_taxa_prenhez": 1.5})


def test_validate_valores_validos_nao_levanta(fitted_prep):
    # Não deve lançar nada
    fitted_prep.validate({
        "condicao_corporal": 3.5,
        "temperatura_ambiente_c": 30.0,
        "historico_taxa_prenhez": 0.75,
        "coeficiente_endogamia": 0.02,
    })


# ---------------------------------------------------------------------------
# Testes de persistência
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(df_sample):
    prep = Preprocessor()
    X_original = prep.fit_transform(df_sample)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    prep.save(path)
    prep2 = Preprocessor.load(path)
    X_carregado = prep2.transform(df_sample)

    assert np.allclose(X_original, X_carregado, atol=1e-6)


def test_transform_sem_fit_levanta_erro():
    prep = Preprocessor()
    with pytest.raises(RuntimeError, match="fit"):
        prep.transform({"condicao_corporal": 3.5})


# ---------------------------------------------------------------------------
# Testes de propriedades
# ---------------------------------------------------------------------------


def test_onehot_categories_populadas(fitted_prep):
    cats = fitted_prep.onehot_categories
    assert "especie" in cats
    assert set(cats["especie"]).issuperset({"BOVINO", "OVINO", "CAPRINO"})
