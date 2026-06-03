"""
Geração de recomendações textuais por templates determinísticos.

Seção 5.6 da Matriz-IA.pdf: recomendações são montadas a partir dos
top-5 fatores determinantes com templates pré-aprovados — nunca LLM.
"""

from __future__ import annotations

from src.schemas import FatorDeterminante

# ---------------------------------------------------------------------------
# Mapa de templates: (feature, sentido) → função que recebe (fator, input_data)
# e retorna string. Usa valores reais do input para personalização.
# ---------------------------------------------------------------------------

def _t(template: str):
    """Helper: cria função de template simples."""
    def render(fator: FatorDeterminante, data: dict) -> str:
        return template.format(valor=fator.valor, **data)
    return render


TEMPLATES: dict[tuple[str, str], callable] = {
    # --- Condição corporal ---
    ("condicao_corporal", "positivo"): lambda f, d: (
        f"Momento favorável para inseminação. A condição corporal está no nível ideal "
        f"({f.valor}/5) e contribui positivamente para o score."
    ),
    ("condicao_corporal", "negativo"): lambda f, d: (
        f"Condição corporal abaixo do ideal ({f.valor}/5). "
        f"Recomenda-se suplementação nutricional por 30–45 dias antes da inseminação."
    ),

    # --- Intervalo pós-parto ---
    ("intervalo_pos_parto_dias", "positivo"): lambda f, d: (
        f"Intervalo pós-parto adequado ({f.valor} dias ≥ 60). "
        f"O trato reprodutivo está recuperado para nova concepção."
    ),
    ("intervalo_pos_parto_dias", "negativo"): lambda f, d: (
        f"Intervalo pós-parto insuficiente ({f.valor} dias < 45). "
        f"Considere aguardar pelo menos 60 dias após o parto para aumentar as chances de prenhez."
    ),

    # --- Histórico taxa prenhez ---
    ("historico_taxa_prenhez", "positivo"): lambda f, d: (
        f"Histórico individual favorável "
        f"({int((f.valor or 0)*100)}% de prenhez em inseminações anteriores). "
        f"Animal com bom desempenho reprodutivo."
    ),
    ("historico_taxa_prenhez", "negativo"): lambda f, d: (
        f"Histórico individual baixo "
        f"({int((f.valor or 0)*100)}% de prenhez). "
        f"Avalie possíveis causas de subfertilidade (nutrição, sanidade, genética)."
    ),

    # --- Temperatura ---
    ("temperatura_ambiente_c", "negativo"): lambda f, d: (
        f"Atenção à temperatura ambiente ({f.valor}°C). "
        f"Realize o procedimento nas horas mais frescas (madrugada ou final da tarde) "
        f"para reduzir o estresse calórico."
    ),

    # --- DEP fertilidade animal ---
    ("dep_fertilidade_animal", "positivo"): lambda f, d: (
        f"DEP de fertilidade do animal favorável ({f.valor}). "
        f"Potencial genético para fertilidade acima da média."
    ),
    ("dep_fertilidade_animal", "negativo"): lambda f, d: (
        f"DEP de fertilidade do animal abaixo de 8.0 (atual: {f.valor}). "
        f"Priorize reprodutores com DEP de fertilidade ≥ 8.0 para compensar."
    ),

    # --- Ciclos sem concepção ---
    ("ciclos_sem_concepcao", "negativo"): lambda f, d: (
        f"Animal com {f.valor} ciclo(s) consecutivo(s) sem concepção. "
        f"Recomenda-se avaliação ginecológica antes de prosseguir com nova inseminação."
    ),

    # --- Coeficiente de endogamia ---
    ("coeficiente_endogamia", "negativo"): lambda f, d: (
        f"Coeficiente de endogamia elevado (F={f.valor:.4f} > 0.0625). "
        f"Use reprodutores de linhagem genética distinta para reduzir consanguinidade."
    ),

    # --- Número de partos ---
    ("num_partos_anteriores", "negativo"): lambda f, d: (
        f"{'Primípara (0 partos anteriores)' if f.valor == 0 else f'Animal com {f.valor} partos'}. "
        f"{'Primíparas tendem a ter taxa de concepção menor na primeira inseminação.' if f.valor == 0 else 'Multíparas com muitos partos têm fertilidade reduzida — avalie descarte ou descanso reprodutivo.'}"
    ),
    ("num_partos_anteriores", "positivo"): lambda f, d: (
        f"Número de partos favorável ({f.valor} partos anteriores). "
        f"Animal com experiência reprodutiva adequada."
    ),

    # --- Tipo inseminação ---
    ("tipo_inseminacao", "positivo"): lambda f, d: (
        f"Protocolo IATF com sincronização hormonal aumenta a taxa de concepção "
        f"em relação à IA convencional (protocolo: {d.get('protocolo_hormonal', 'não informado')})."
    ),

    # --- Dias desde última inseminação ---
    ("dias_desde_ultima_ins", "negativo"): lambda f, d: (
        f"Inseminação proposta antes do ciclo completo ({f.valor} dias). "
        f"Aguarde pelo menos {'21' if d.get('especie') == 'BOVINO' else '17'} dias para BOVINO/O/C."
    ),
}

# Recomendações genéricas por classificação (fallback)
_FALLBACK: dict[str, list[str]] = {
    "FAVORAVEL": [
        "Condições gerais favoráveis. Prossiga com a inseminação conforme protocolo estabelecido.",
        "Monitore a evolução da gestação com diagnóstico aos 28–35 dias pós-inseminação.",
        "Registre o resultado no sistema para realimentar o modelo de predição.",
    ],
    "MEDIO": [
        "Há fatores de atenção. Revise as condições antes de prosseguir.",
        "Considere adiar a inseminação se alguma condição puder ser melhorada em 2–4 semanas.",
        "Registre o resultado independentemente do desfecho — dados reais melhoram o modelo.",
    ],
    "DESFAVORAVEL": [
        "Risco elevado. Avalie adiar a inseminação e corrigir os fatores críticos primeiro.",
        "Consulte um médico veterinário para avaliação ginecológica antes de prosseguir.",
        "Melhore a condição corporal, respeite o intervalo pós-parto e revise os DEPs do reprodutor.",
    ],
}


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------


def gerar_recomendacoes(
    fatores: list[FatorDeterminante],
    input_data: dict,
    classificacao: str = "MEDIO",
    max_rec: int = 3,
) -> list[str]:
    """
    Gera até max_rec recomendações textuais a partir dos top-5 fatores.

    Seleciona templates com base em (feature, sentido) de cada fator.
    Completa com fallbacks por classificação se necessário.
    """
    recomendacoes: list[str] = []
    usadas: set[str] = set()

    for fator in fatores:
        if len(recomendacoes) >= max_rec:
            break
        chave = (fator.feature, fator.sentido)
        fn = TEMPLATES.get(chave)
        if fn is None:
            continue
        try:
            texto = fn(fator, input_data)
        except Exception:
            continue
        if texto not in usadas:
            recomendacoes.append(texto)
            usadas.add(texto)

    # Completa com fallbacks se necessário
    for fb in _FALLBACK.get(classificacao, _FALLBACK["MEDIO"]):
        if len(recomendacoes) >= max_rec:
            break
        if fb not in usadas:
            recomendacoes.append(fb)
            usadas.add(fb)

    return recomendacoes[:max_rec]
