"""
Gerador do dataset de cold start para o AgroGen IA.

Gera 1.300 registros sintéticos calibrados por literatura zootécnica
(EMBRAPA GENECOC, Hafez 2004). Usa distribuições condicionais por classe
(prenha=1 / prenha=0) para garantir separabilidade suficiente ao treino
do Random Forest (AUC-ROC alvo > 0.80).

Cada linha representa uma inseminação histórica com resultado confirmado
por diagnóstico de gestação.

Saída: data/cold_start_v1.csv
  - 14 features ativas (entrada do modelo ML)
  - 3 extras (motor de regras)
  - 13 features auxiliares (descartadas no MVP)
  - 1 target binário (prenha: 1 = PRENHA, 0 = VAZIA)

Taxas base por espécie: BOVINO 60%, OVINO 65%, CAPRINO 65%.
"""

import hashlib
import os
import random

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_TOTAL = 1300
N_BOVINO = 500
N_OVINO = 400
N_CAPRINO = 400

PROB_BASE = {"BOVINO": 0.60, "OVINO": 0.65, "CAPRINO": 0.65}

RACAS = {
    "BOVINO": ["Nelore", "Angus", "Brahman", "Gir", "Guzerá", "Tabapuã", "Simmental"],
    "OVINO": ["Santa Inês", "Morada Nova", "Dorper", "Suffolk", "Somalis Brasileira"],
    "CAPRINO": ["Anglo-nubiano", "Moxotó", "Boer", "Saanen"],
}

RACAS_ADAPTADAS = {"Nelore", "Santa Inês", "Anglo-nubiano", "Moxotó"}

PROTOCOLOS_IATF = ["P4+EB 7 dias", "OvSynch", "Crestar", "CIDR+GnRH"]

CICLO_MINIMO = {"BOVINO": 21, "OVINO": 17, "CAPRINO": 17}


def _rng_choice(vals, p):
    return vals[np.random.choice(len(vals), p=p)]


def _gerar_registro(especie: str, target: int) -> dict:
    """
    Gera um registro com features amostradas de distribuições condicionais
    ao resultado (target=1 → PRENHA, target=0 → VAZIA).

    Distribuições favoráveis para PRENHA, desfavoráveis para VAZIA, com
    ~15% de sobreposição para manter realismo.
    """
    r: dict = {}
    racas = RACAS[especie]
    ciclo = CICLO_MINIMO[especie]

    r["especie"] = especie
    r["raca_femea"] = random.choice(racas)

    # ----- Condição corporal -----
    if target == 1:
        # CC próxima do ideal (3-4)
        r["condicao_corporal"] = float(np.clip(
            round(np.random.normal(3.5, 0.4) * 2) / 2, 2.5, 4.5
        ))
    else:
        # CC distribuída em extremos ou média baixa
        group = np.random.choice(["baixa", "alta", "media"], p=[0.45, 0.20, 0.35])
        if group == "baixa":
            cc = np.random.normal(2.0, 0.35)
        elif group == "alta":
            cc = np.random.normal(4.8, 0.25)
        else:
            cc = np.random.normal(3.0, 0.5)
        r["condicao_corporal"] = float(np.clip(round(cc * 2) / 2, 1.0, 5.0))

    # ----- Intervalo pós-parto -----
    if target == 1:
        r["intervalo_pos_parto_dias"] = int(np.clip(
            np.random.normal(80, 18), 45, 300
        ))
    else:
        r["intervalo_pos_parto_dias"] = int(np.clip(
            np.random.normal(42, 15), 15, 150
        ))

    # ----- Número de partos anteriores -----
    if target == 1:
        # Favorece multíparas com boa experiência (2-4 partos)
        r["num_partos_anteriores"] = _rng_choice(
            [0, 1, 2, 3, 4, 5, 6],
            [0.05, 0.15, 0.28, 0.25, 0.15, 0.08, 0.04],
        )
    else:
        # Favorece primíparas e muito multíparas
        r["num_partos_anteriores"] = _rng_choice(
            [0, 1, 2, 3, 4, 5, 7, 8],
            [0.30, 0.20, 0.12, 0.10, 0.08, 0.06, 0.08, 0.06],
        )

    # ----- Histórico taxa de prenhez -----
    if target == 1:
        r["historico_taxa_prenhez"] = round(
            np.clip(np.random.normal(0.72, 0.12), 0.40, 1.0), 2
        )
    else:
        r["historico_taxa_prenhez"] = round(
            np.clip(np.random.normal(0.38, 0.15), 0.0, 0.70), 2
        )

    # ----- Ciclos sem concepção (extra — motor de regras) -----
    if target == 1:
        r["ciclos_sem_concepcao"] = _rng_choice([0, 1, 2], [0.60, 0.30, 0.10])
    else:
        r["ciclos_sem_concepcao"] = _rng_choice([0, 1, 2, 3, 4], [0.20, 0.25, 0.25, 0.18, 0.12])

    # ----- Dias desde última inseminação -----
    if target == 1:
        # Respeita o ciclo da espécie
        r["dias_desde_ultima_ins"] = random.randint(ciclo, 180)
    else:
        # 20% inseminam antes do ciclo completo
        if random.random() < 0.20:
            r["dias_desde_ultima_ins"] = random.randint(1, ciclo - 1)
        else:
            r["dias_desde_ultima_ins"] = random.randint(ciclo, 180)

    # ----- DEP fertilidade animal -----
    if target == 1:
        r["dep_fertilidade_animal"] = round(
            np.clip(np.random.normal(8.0, 2.0), 3.0, 15.0), 1
        )
    else:
        r["dep_fertilidade_animal"] = round(
            np.clip(np.random.normal(4.5, 2.5), 0.0, 12.0), 1
        )

    # ----- DEP fertilidade reprodutor (extra — motor de regras) -----
    if target == 1:
        r["dep_fertilidade_reprodutor"] = round(
            np.clip(np.random.normal(7.5, 2.0), 3.0, 15.0), 1
        )
    else:
        r["dep_fertilidade_reprodutor"] = round(
            np.clip(np.random.normal(4.0, 2.5), 0.0, 12.0), 1
        )

    # ----- DEP acurácia -----
    r["dep_acuracia"] = round(
        np.clip(np.random.beta(5 if target == 1 else 3, 2), 0.10, 0.99), 2
    )

    # ----- Coeficiente de endogamia -----
    if target == 1:
        # Quase sempre baixo
        r["coeficiente_endogamia"] = round(
            np.clip(np.random.beta(1, 25), 0.0, 0.06), 4
        )
    else:
        # 12% com endogamia elevada
        if random.random() < 0.12:
            r["coeficiente_endogamia"] = round(
                np.clip(np.random.uniform(0.0625, 0.25), 0.0, 0.30), 4
            )
        else:
            r["coeficiente_endogamia"] = round(
                np.clip(np.random.beta(1, 20), 0.0, 0.06), 4
            )

    # ----- Heterose esperada (extra — motor de regras) -----
    if target == 1:
        r["heterose_esperada"] = round(
            np.clip(np.random.normal(4.0, 2.0), 0.0, 12.0), 1
        )
    else:
        r["heterose_esperada"] = round(
            np.clip(np.random.normal(1.5, 2.0), 0.0, 8.0), 1
        )

    # ----- Tipo de inseminação e protocolo -----
    prob_iatf = 0.75 if target == 1 else 0.50
    if random.random() < prob_iatf:
        r["tipo_inseminacao"] = "IATF"
        r["protocolo_hormonal"] = random.choice(PROTOCOLOS_IATF)
    else:
        r["tipo_inseminacao"] = "IA_CONVENCIONAL"
        r["protocolo_hormonal"] = "IA_CONVENCIONAL"

    # ----- Estação e temperatura -----
    r["estacao"] = random.choice(["SECA", "CHUVOSA"])
    if target == 1:
        # Temperaturas mais amenas
        base = 26.0 if r["estacao"] == "CHUVOSA" else 27.5
        r["temperatura_ambiente_c"] = round(
            np.clip(np.random.normal(base, 3.5), 18.0, 35.0), 1
        )
    else:
        # Estresse calórico mais frequente
        base = 27.0 if r["estacao"] == "CHUVOSA" else 31.0
        r["temperatura_ambiente_c"] = round(
            np.clip(np.random.normal(base, 4.5), 18.0, 45.0), 1
        )

    r["prenha"] = target

    # ----- Features auxiliares (descartadas no MVP) -----
    r["peso_atual_kg"] = round(
        np.clip(np.random.normal(400 if especie == "BOVINO" else 60, 50), 100, 800), 1
    )
    r["idade_meses"] = max(12, r["num_partos_anteriores"] * 14 + random.randint(12, 24))
    r["pluviometria_mm"] = round(
        np.random.exponential(80) if r["estacao"] == "CHUVOSA" else np.random.exponential(15), 1
    )
    r["umidade_relativa"] = round(
        np.clip(np.random.normal(65 if r["estacao"] == "CHUVOSA" else 45, 15), 20, 100), 1
    )
    r["tecnico_id"] = f"TEC-{random.randint(1, 8):03d}"
    r["escore_locomotor"] = _rng_choice([1, 2, 3, 4, 5], [0.40, 0.30, 0.20, 0.07, 0.03])
    r["num_inseminacoes_total"] = max(1, r["num_partos_anteriores"] + random.randint(0, 3))
    r["raca_reprodutor"] = random.choice(["Nelore", "Angus", "Brahman", "Simental", "Gir"])
    r["dep_peso_desmame"] = round(np.clip(np.random.normal(12.0, 5.0), -5.0, 30.0), 1)
    r["dep_ganho_pos_desmame"] = round(np.clip(np.random.normal(10.0, 4.0), -5.0, 25.0), 1)
    r["ciclo_estral_dias"] = ciclo
    r["resultado_diagnostico_anterior"] = _rng_choice(
        ["PRENHA", "VAZIA", "PRIMEIRA_VEZ"],
        [0.55 if target == 1 else 0.30, 0.35 if target == 1 else 0.60, 0.10],
    )
    r["intervalo_entre_partos_dias"] = (
        int(np.clip(np.random.normal(380, 40), 280, 600))
        if r["num_partos_anteriores"] >= 2
        else None
    )

    return r


def gerar_registros_especie(especie: str, n: int) -> list[dict]:
    """Gera n registros para uma espécie com a taxa de prenhez alvo."""
    prob_prenha = PROB_BASE[especie]
    n_prenha = round(n * prob_prenha)
    n_vazia = n - n_prenha

    registros = (
        [_gerar_registro(especie, 1) for _ in range(n_prenha)]
        + [_gerar_registro(especie, 0) for _ in range(n_vazia)]
    )
    # Embaralha para que treino/teste não vejam padrões de posição
    random.shuffle(registros)
    return registros


def main() -> None:
    registros: list[dict] = []
    registros.extend(gerar_registros_especie("BOVINO", N_BOVINO))
    registros.extend(gerar_registros_especie("OVINO", N_OVINO))
    registros.extend(gerar_registros_especie("CAPRINO", N_CAPRINO))

    df = pd.DataFrame(registros)

    features_ml = [
        "condicao_corporal", "historico_taxa_prenhez", "intervalo_pos_parto_dias",
        "num_partos_anteriores", "dias_desde_ultima_ins", "dep_fertilidade_animal",
        "especie", "raca_femea", "tipo_inseminacao", "protocolo_hormonal",
        "temperatura_ambiente_c", "estacao", "dep_acuracia", "coeficiente_endogamia",
    ]
    extras_regras = ["ciclos_sem_concepcao", "dep_fertilidade_reprodutor", "heterose_esperada"]
    auxiliares = [
        "peso_atual_kg", "idade_meses", "pluviometria_mm", "umidade_relativa",
        "tecnico_id", "escore_locomotor", "num_inseminacoes_total", "raca_reprodutor",
        "dep_peso_desmame", "dep_ganho_pos_desmame", "ciclo_estral_dias",
        "resultado_diagnostico_anterior", "intervalo_entre_partos_dias",
    ]
    df = df[features_ml + extras_regras + auxiliares + ["prenha"]]

    output_path = os.path.join(os.path.dirname(__file__), "cold_start_v1.csv")
    df.to_csv(output_path, index=False)

    taxa_geral = df["prenha"].mean()
    print(f"Registros gerados: {len(df)}")
    print(f"Taxa de prenhez geral: {taxa_geral:.3f} (alvo ~0.62)")
    for esp in ["BOVINO", "OVINO", "CAPRINO"]:
        sub = df[df["especie"] == esp]
        print(f"  {esp}: {len(sub)} registros, taxa = {sub['prenha'].mean():.3f}")

    sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"\nArquivo: {output_path}\nSHA-256: {sha256}")
    print(f"Colunas: {len(df.columns)} ({len(features_ml)} ML + {len(extras_regras)} regras + {len(auxiliares)} aux + 1 target)")


if __name__ == "__main__":
    main()
