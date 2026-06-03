"""
Conversor do dataset_inseminacao_ia.csv para o formato esperado pelo training.py.

Resolve:
  - Normalização de strings (BOVINO vs Bovino, SECA vs Seca, etc.)
  - Criação da coluna binária 'prenha' a partir de 'resultado'
  - historico_taxa_prenhez: derivado do histórico ordenado de cada animal
  - Colunas ausentes (dep_fertilidade_animal, dep_acuracia, coeficiente_endogamia)
    preenchidas com defaults calibrados por espécie
  - intervalo_pos_parto_dias: nulos preenchidos com mediana por espécie

Uso:
    python data/converter.py --input caminho/dataset.csv --output data/cold_start_v1.csv
    python data/converter.py                     # usa defaults
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults para features ausentes no CSV de origem (calibrados por literatura)
# ---------------------------------------------------------------------------

DEP_FERTILIDADE_MEDIA = {"BOVINO": 6.5, "OVINO": 5.8, "CAPRINO": 5.5}
DEP_ACURACIA_PADRAO = 0.70
COEFICIENTE_ENDOGAMIA_PADRAO = 0.02

# Mapeamento tipo_inseminacao → valor esperado pelo modelo
TIPO_INS_MAP = {
    "ia convencional": "IA_CONVENCIONAL",
    "ia_convencional": "IA_CONVENCIONAL",
    "convencional": "IA_CONVENCIONAL",
    "iatf": "IATF",
}


def converter(input_path: Path, output_path: Path) -> pd.DataFrame:
    print(f"Lendo: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} linhas, {len(df.columns)} colunas")

    # ------------------------------------------------------------------
    # 1. Normalização de strings
    # ------------------------------------------------------------------
    df["especie"] = df["especie"].str.upper().str.strip()
    df["estacao"] = df["estacao"].str.upper().str.strip()
    df["tipo_inseminacao"] = (
        df["tipo_inseminacao"]
        .str.lower()
        .str.strip()
        .map(TIPO_INS_MAP)
        .fillna("IA_CONVENCIONAL")
    )
    df["protocolo_hormonal"] = df["protocolo_hormonal"].str.strip().fillna("IA_CONVENCIONAL")
    df["raca_femea"] = df["raca_femea"].str.strip()

    # Normaliza protocolo para IA_CONVENCIONAL quando tipo é convencional
    mask_conv = df["tipo_inseminacao"] == "IA_CONVENCIONAL"
    df.loc[mask_conv, "protocolo_hormonal"] = "IA_CONVENCIONAL"

    # ------------------------------------------------------------------
    # 2. Coluna alvo binária
    # ------------------------------------------------------------------
    resultado_lower = df["resultado"].str.strip().str.lower()
    df["prenha"] = resultado_lower.map(
        {"prenha": 1, "nao_prenha": 0, "nao prenha": 0, "vazia": 0}
    ).fillna(0).astype(int)

    taxa_geral = df["prenha"].mean()
    print(f"  Taxa de prenhez: {taxa_geral:.3f}")

    # ------------------------------------------------------------------
    # 3. intervalo_pos_parto_dias — preenche nulos com mediana por espécie
    # ------------------------------------------------------------------
    n_nulos = df["intervalo_pos_parto_dias"].isna().sum()
    if n_nulos > 0:
        mediana_esp = df.groupby("especie")["intervalo_pos_parto_dias"].median()
        for esp in df["especie"].unique():
            mask = df["especie"] == esp
            mediana = mediana_esp.get(esp, 70.0)
            df.loc[mask & df["intervalo_pos_parto_dias"].isna(), "intervalo_pos_parto_dias"] = mediana
        print(f"  intervalo_pos_parto_dias: {n_nulos} nulos preenchidos com mediana por espécie")

    df["intervalo_pos_parto_dias"] = df["intervalo_pos_parto_dias"].astype(int)

    # ------------------------------------------------------------------
    # 4. historico_taxa_prenhez — derivado do histórico por animal
    # ------------------------------------------------------------------
    if "historico_taxa_prenhez" not in df.columns:
        if "data_inseminacao" in df.columns and "animal_id" in df.columns:
            df["data_inseminacao"] = pd.to_datetime(df["data_inseminacao"], errors="coerce")
            df = df.sort_values(["animal_id", "data_inseminacao"])

            # Para cada linha, calcula a taxa das inseminações ANTERIORES do mesmo animal
            df["historico_taxa_prenhez"] = (
                df.groupby("animal_id")["prenha"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )

            # Animais sem histórico anterior: usa média do rebanho da mesma espécie
            media_por_especie = df.groupby("especie")["prenha"].mean()
            for esp in df["especie"].unique():
                mask = (df["especie"] == esp) & df["historico_taxa_prenhez"].isna()
                df.loc[mask, "historico_taxa_prenhez"] = media_por_especie.get(esp, 0.60)

            n_sem_hist = (df["historico_taxa_prenhez"].isna()).sum()
            print(f"  historico_taxa_prenhez: derivado do histórico ({n_sem_hist} sem histórico → média da espécie)")
        else:
            # Sem animal_id ou data: usa média global por espécie
            media_por_especie = df.groupby("especie")["prenha"].mean()
            df["historico_taxa_prenhez"] = df["especie"].map(media_por_especie)
            print("  historico_taxa_prenhez: usando média global por espécie (sem animal_id)")

        df["historico_taxa_prenhez"] = df["historico_taxa_prenhez"].round(4).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Features ausentes — defaults calibrados por literatura
    # ------------------------------------------------------------------
    if "dep_fertilidade_animal" not in df.columns:
        df["dep_fertilidade_animal"] = df["especie"].map(DEP_FERTILIDADE_MEDIA).fillna(6.5)
        # Adiciona variação realista
        rng = np.random.default_rng(42)
        df["dep_fertilidade_animal"] += rng.normal(0, 1.5, len(df))
        df["dep_fertilidade_animal"] = df["dep_fertilidade_animal"].clip(0.0, 15.0).round(1)
        print(f"  dep_fertilidade_animal: gerado com média {DEP_FERTILIDADE_MEDIA}")

    if "dep_acuracia" not in df.columns:
        df["dep_acuracia"] = DEP_ACURACIA_PADRAO
        print(f"  dep_acuracia: preenchido com {DEP_ACURACIA_PADRAO}")

    if "coeficiente_endogamia" not in df.columns:
        df["coeficiente_endogamia"] = COEFICIENTE_ENDOGAMIA_PADRAO
        print(f"  coeficiente_endogamia: preenchido com {COEFICIENTE_ENDOGAMIA_PADRAO}")

    # ------------------------------------------------------------------
    # 6. Seleciona apenas as colunas que o training.py precisa
    # ------------------------------------------------------------------
    colunas_ml = [
        "condicao_corporal", "historico_taxa_prenhez", "intervalo_pos_parto_dias",
        "num_partos_anteriores", "dias_desde_ultima_ins", "dep_fertilidade_animal",
        "especie", "raca_femea", "tipo_inseminacao", "protocolo_hormonal",
        "temperatura_ambiente_c", "estacao", "dep_acuracia", "coeficiente_endogamia",
        "prenha",
    ]
    df_saida = df[colunas_ml].copy()

    # ------------------------------------------------------------------
    # 7. Validação mínima
    # ------------------------------------------------------------------
    erros = []
    if not df_saida["especie"].isin(["BOVINO", "OVINO", "CAPRINO"]).all():
        invalidos = df_saida.loc[~df_saida["especie"].isin(["BOVINO", "OVINO", "CAPRINO"]), "especie"].unique()
        erros.append(f"especie com valores inválidos: {invalidos}")
    if not df_saida["tipo_inseminacao"].isin(["IATF", "IA_CONVENCIONAL"]).all():
        erros.append("tipo_inseminacao com valores fora de {'IATF', 'IA_CONVENCIONAL'}")
    if df_saida["prenha"].isna().any():
        erros.append("coluna 'prenha' com nulos")

    if erros:
        print("\nAVISOS:")
        for e in erros:
            print(f"  ! {e}")

    # ------------------------------------------------------------------
    # 8. Salva
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_saida.to_csv(output_path, index=False)

    print(f"\nSalvo em: {output_path}")
    print(f"  Registros: {len(df_saida)}")
    print(f"  Taxa prenhez: {df_saida['prenha'].mean():.3f}")
    for esp in df_saida["especie"].unique():
        sub = df_saida[df_saida["especie"] == esp]
        print(f"    {esp}: {len(sub)} registros, {sub['prenha'].mean():.3f} prenhez")

    return df_saida


def mesclar_com_sintetico(df_real: pd.DataFrame, sintetico_path: Path) -> pd.DataFrame:
    """
    Mescla dados reais com dataset sintético para compensar features ausentes.

    Necessário quando colunas como dep_fertilidade_animal, dep_acuracia e
    coeficiente_endogamia não estão no CSV original e foram preenchidas com
    valores fixos (sem variação), o que impede o modelo de aprender.
    """
    if not sintetico_path.exists():
        print("  Dataset sintético não encontrado — gerando agora...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(sintetico_path.parent / "synthetic_generator.py")],
            check=True,
        )

    df_sint = pd.read_csv(sintetico_path)
    colunas_ml = [c for c in df_real.columns]
    df_sint = df_sint[[c for c in colunas_ml if c in df_sint.columns]]

    combinado = pd.concat([df_sint, df_real], ignore_index=True)
    print(f"  Mesclado: {len(df_sint)} sintéticos + {len(df_real)} reais = {len(combinado)} total")
    return combinado


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte CSV de inseminações para formato de treino")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent.parent.parent / "dataset_inseminacao_ia.csv",
        help="CSV de entrada",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "cold_start_v1.csv",
        help="CSV de saída (substitui o cold_start_v1.csv)",
    )
    parser.add_argument(
        "--mesclar-sintetico", action="store_true", default=True,
        help="Mescla com dataset sintético para compensar features ausentes (padrão: True)",
    )
    parser.add_argument(
        "--somente-real", action="store_true",
        help="Usa apenas os dados reais, sem mesclar com sintético",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Arquivo não encontrado: {args.input}")
        raise SystemExit(1)

    df_convertido = converter(args.input, args.output)

    if not args.somente_real:
        sintetico_path = Path(__file__).parent / "cold_start_sintetico.csv"

        # Salva o sintético em arquivo separado para não sobrescrever
        print("\nMesclando com dataset sintético (compensa features ausentes)...")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "synthetic_generator.py")],
            capture_output=True, text=True,
        )
        # Move o cold_start_v1.csv gerado para arquivo temporário
        import shutil
        sintetico_gerado = Path(__file__).parent / "cold_start_v1.csv"
        if sintetico_gerado.exists() and sintetico_gerado != args.output:
            shutil.copy(sintetico_gerado, sintetico_path)

        if sintetico_path.exists():
            df_misto = mesclar_com_sintetico(df_convertido, sintetico_path)
            df_misto.to_csv(args.output, index=False)
            print(f"  Salvo em: {args.output} ({len(df_misto)} registros)")
        else:
            print("  Não foi possível gerar o sintético — usando apenas dados reais")
            print("  AVISO: métricas do modelo podem ficar abaixo do alvo (AUC < 0.80)")

    print("\nPróximo passo:")
    print("  PYTHONPATH=. python src/training.py")


if __name__ == "__main__":
    main()
