"""
Gerador do dataset de cold start para o AgroGen IA.

Gera 1.300 registros sintéticos calibrados por literatura zootécnica
(EMBRAPA GENECOC, Hafez 2004). Cada linha representa uma inseminação
histórica com resultado confirmado por diagnóstico de gestação.

Saída: data/cold_start_v1.csv
  - 14 features ativas (entrada do modelo ML)
  - 13 features auxiliares (descartadas no MVP)
  - 1 target binário (prenha: 1 = PRENHA, 0 = VAZIA)

Taxas base por espécie: BOVINO 60%, OVINO 65%, CAPRINO 65%.
"""

import hashlib
import math
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

PROTOCOLOS_IATF = [
    "P4+EB 7 dias",
    "OvSynch",
    "Crestar",
    "CIDR+GnRH",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# Offset de calibração por espécie: ajusta a distribuição sintética para
# que a taxa marginal de prenhez fique próxima às taxas base reais.
# Valor negativo reduz a probabilidade média (distribuição sintética é otimista).
CALIBRACAO_OFFSET = {"BOVINO": -0.55, "OVINO": -0.40, "CAPRINO": -0.40}


def calcular_score_latente(row: dict) -> float:
    """
    Calcula probabilidade latente de prenhez baseada nos deltas da seção 3.3.
    Usa a fórmula sigmoide: score = sigmoid(Σdeltas / 40 + logit(prob_base) + calibracao).
    """
    deltas = 0.0

    # Condição corporal
    cc = row["condicao_corporal"]
    if 3.0 <= cc <= 4.0:
        deltas += 10
    elif cc in (2.5, 4.5) or (cc == 2.5 or cc == 4.5):
        deltas += 3
    elif cc < 2.5 or cc > 4.5:
        deltas -= 12

    # Intervalo pós-parto
    ipp = row["intervalo_pos_parto_dias"]
    if ipp >= 60:
        deltas += 4
    elif ipp < 45:
        deltas -= 15

    # Número de partos
    np_ant = row["num_partos_anteriores"]
    if np_ant == 0:
        deltas -= 4
    elif 2 <= np_ant <= 4:
        deltas += 6
    elif np_ant >= 7:
        deltas -= 5

    # Histórico taxa prenhez
    taxa = row["historico_taxa_prenhez"]
    if taxa >= 0.70:
        deltas += 8
    elif taxa < 0.40:
        deltas -= 7

    # Dias desde última inseminação
    dias_ins = row["dias_desde_ultima_ins"]
    ciclo = 21 if row["especie"] == "BOVINO" else 17
    if dias_ins < ciclo:
        deltas -= 10

    # Tipo inseminação
    if row["tipo_inseminacao"] == "IATF":
        deltas += 5

    # Temperatura
    temp = row["temperatura_ambiente_c"]
    if 29 <= temp <= 33:
        deltas -= 2
    elif temp >= 34:
        deltas -= 8

    # Raça adaptada
    if row["raca_femea"] in RACAS_ADAPTADAS:
        deltas += 4

    # Heterose
    if row.get("heterose_esperada", 0) >= 4.0:
        deltas += 4

    # Endogamia
    if row["coeficiente_endogamia"] > 0.0625:
        deltas -= 10

    # DEP fertilidade somada
    dep_sum = row["dep_fertilidade_animal"] + row.get("dep_fertilidade_reprodutor", 0.0)
    dep_delta = 6.0 if dep_sum >= 12 else 0.0
    # Aplica metade do delta se acurácia baixa
    if row["dep_acuracia"] < 0.40:
        dep_delta /= 2.0
    deltas += dep_delta

    # Ciclos sem concepção (extra, não entra no ML)
    ciclos = row.get("ciclos_sem_concepcao", 0)
    if ciclos >= 3:
        deltas -= 6
    elif ciclos == 2:
        deltas -= 3

    offset = logit(PROB_BASE[row["especie"]])
    calibracao = CALIBRACAO_OFFSET[row["especie"]]
    score = sigmoid(deltas / 40.0 + offset + calibracao)
    return float(np.clip(score, 0.05, 0.95))


def gerar_registros_especie(especie: str, n: int) -> list[dict]:
    racas = RACAS[especie]
    registros = []

    for _ in range(n):
        r: dict = {}

        r["especie"] = especie
        r["raca_femea"] = random.choice(racas)

        # Número de partos (distribuição realista)
        num_partos_weights = [0.15, 0.20, 0.25, 0.20, 0.10, 0.06, 0.04]
        r["num_partos_anteriores"] = random.choices(range(7), weights=num_partos_weights)[0]

        # Condição corporal — correlacionada com num_partos
        # Animais sem partos e com muitos partos tendem a ter CC menor
        if r["num_partos_anteriores"] == 0:
            cc_mean = 3.0
        elif r["num_partos_anteriores"] >= 6:
            cc_mean = 2.8
        else:
            cc_mean = 3.4
        cc_raw = np.random.normal(cc_mean, 0.6)
        r["condicao_corporal"] = float(np.clip(round(cc_raw * 2) / 2, 1.0, 5.0))  # resolução 0.5

        # Intervalo pós-parto — correlacionado com CC
        if r["num_partos_anteriores"] == 0:
            r["intervalo_pos_parto_dias"] = int(np.clip(np.random.normal(120, 30), 30, 400))
        else:
            # CC baixa → IPP curto (produtores não esperam)
            ipp_base = 45 + 40 * (r["condicao_corporal"] - 1) / 4
            r["intervalo_pos_parto_dias"] = int(np.clip(np.random.normal(ipp_base, 20), 20, 300))

        # Histórico taxa prenhez — primíparas têm histórico neutro
        if r["num_partos_anteriores"] == 0:
            r["historico_taxa_prenhez"] = round(np.clip(np.random.normal(0.55, 0.10), 0.0, 1.0), 2)
        else:
            r["historico_taxa_prenhez"] = round(
                np.clip(np.random.normal(0.62, 0.18), 0.0, 1.0), 2
            )

        # Dias desde última inseminação
        ciclo = 21 if especie == "BOVINO" else 17
        # Maioria respeita o ciclo, alguns não
        if random.random() < 0.12:
            r["dias_desde_ultima_ins"] = random.randint(1, ciclo - 1)
        else:
            r["dias_desde_ultima_ins"] = random.randint(ciclo, 180)

        # Ciclos sem concepção (extra — motor de regras)
        if r["historico_taxa_prenhez"] < 0.40:
            r["ciclos_sem_concepcao"] = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0]
        else:
            r["ciclos_sem_concepcao"] = random.choices([0, 1, 2, 3], weights=[0.55, 0.30, 0.10, 0.05])[0]

        # DEP fertilidade animal
        r["dep_fertilidade_animal"] = round(np.clip(np.random.normal(6.5, 2.5), 0.0, 15.0), 1)

        # DEP fertilidade reprodutor (extra — motor de regras)
        r["dep_fertilidade_reprodutor"] = round(np.clip(np.random.normal(6.0, 2.5), 0.0, 15.0), 1)

        # DEP acurácia
        r["dep_acuracia"] = round(np.clip(np.random.beta(5, 2), 0.10, 0.99), 2)

        # Coeficiente de endogamia — maioria baixo, ~5% elevado
        if random.random() < 0.05:
            r["coeficiente_endogamia"] = round(np.clip(np.random.uniform(0.0625, 0.25), 0.0, 0.30), 4)
        else:
            r["coeficiente_endogamia"] = round(np.clip(np.random.beta(1, 20), 0.0, 0.06), 4)

        # Heterose esperada (extra — motor de regras)
        # Maior quando raça é adaptada × exótica
        if r["raca_femea"] in RACAS_ADAPTADAS:
            r["heterose_esperada"] = round(np.clip(np.random.normal(3.5, 2.0), 0.0, 12.0), 1)
        else:
            r["heterose_esperada"] = round(np.clip(np.random.normal(1.5, 1.5), 0.0, 8.0), 1)

        # Tipo inseminação e protocolo
        if random.random() < 0.65:
            r["tipo_inseminacao"] = "IATF"
            r["protocolo_hormonal"] = random.choice(PROTOCOLOS_IATF)
        else:
            r["tipo_inseminacao"] = "IA_CONVENCIONAL"
            r["protocolo_hormonal"] = "IA_CONVENCIONAL"

        # Estação e temperatura — correlacionados
        r["estacao"] = random.choice(["SECA", "CHUVOSA"])
        if r["estacao"] == "SECA":
            # CC baixa → mais exposta ao calor
            temp_base = 30.0 if r["condicao_corporal"] < 2.5 else 28.5
            r["temperatura_ambiente_c"] = round(np.clip(np.random.normal(temp_base, 4.0), 15.0, 45.0), 1)
        else:
            r["temperatura_ambiente_c"] = round(np.clip(np.random.normal(26.0, 3.5), 15.0, 40.0), 1)

        # --- Features auxiliares (descartadas no MVP) ---
        r["peso_atual_kg"] = round(np.clip(np.random.normal(400 if especie == "BOVINO" else 60, 50), 100, 800), 1)
        r["idade_meses"] = max(12, r["num_partos_anteriores"] * 14 + random.randint(12, 24))
        r["pluviometria_mm"] = round(np.random.exponential(80) if r["estacao"] == "CHUVOSA" else np.random.exponential(15), 1)
        r["umidade_relativa"] = round(np.clip(np.random.normal(65 if r["estacao"] == "CHUVOSA" else 45, 15), 20, 100), 1)
        r["tecnico_id"] = f"TEC-{random.randint(1, 8):03d}"
        r["escore_locomotor"] = random.choices([1, 2, 3, 4, 5], weights=[0.40, 0.30, 0.20, 0.07, 0.03])[0]
        r["num_inseminacoes_total"] = max(1, r["num_partos_anteriores"] + random.randint(0, 3))
        r["raca_reprodutor"] = random.choice(["Nelore", "Angus", "Brahman", "Simental", "Gir"])
        r["dep_peso_desmame"] = round(np.clip(np.random.normal(12.0, 5.0), -5.0, 30.0), 1)
        r["dep_ganho_pos_desmame"] = round(np.clip(np.random.normal(10.0, 4.0), -5.0, 25.0), 1)
        r["ciclo_estral_dias"] = 21 if especie == "BOVINO" else 17
        r["resultado_diagnostico_anterior"] = random.choices(
            ["PRENHA", "VAZIA", "PRIMEIRA_VEZ"],
            weights=[0.55, 0.35, 0.10],
        )[0]
        r["intervalo_entre_partos_dias"] = (
            int(np.clip(np.random.normal(380, 40), 280, 600))
            if r["num_partos_anteriores"] >= 2
            else None
        )

        # --- Target ---
        p_latente = calcular_score_latente(r)
        # Adiciona ruído para realismo (um preditor não captura tudo)
        p_com_ruido = np.clip(p_latente + np.random.normal(0, 0.08), 0.05, 0.95)
        r["prenha"] = int(np.random.binomial(1, p_com_ruido))

        registros.append(r)

    return registros


def main() -> None:
    registros: list[dict] = []
    registros.extend(gerar_registros_especie("BOVINO", N_BOVINO))
    registros.extend(gerar_registros_especie("OVINO", N_OVINO))
    registros.extend(gerar_registros_especie("CAPRINO", N_CAPRINO))

    df = pd.DataFrame(registros)

    # Ordem das colunas: 14 features ML | extras motor de regras | auxiliares | target
    features_ml = [
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

    # Estatísticas de validação
    taxa_geral = df["prenha"].mean()
    print(f"Registros gerados: {len(df)}")
    print(f"Taxa de prenhez geral: {taxa_geral:.3f} (alvo ~0.62)")
    for esp in ["BOVINO", "OVINO", "CAPRINO"]:
        sub = df[df["especie"] == esp]
        print(f"  {esp}: {len(sub)} registros, taxa prenhez = {sub['prenha'].mean():.3f} (alvo {PROB_BASE[esp]:.2f})")

    sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"\nArquivo: {output_path}")
    print(f"SHA-256: {sha256}")
    print(f"Colunas: {len(df.columns)} ({len(features_ml)} ML + {len(extras_regras)} regras + {len(auxiliares)} aux + 1 target)")


if __name__ == "__main__":
    main()
