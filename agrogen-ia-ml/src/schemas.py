"""
Schemas Pydantic v2 para os endpoints do microsserviço AgroGen IA.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# POST /predicao — Request
# ---------------------------------------------------------------------------


class PredicaoRequest(BaseModel):
    """
    Vetor de features montado pelo backend Java e enviado ao microsserviço.

    Os 14 primeiros campos são as features canônicas do modelo ML.
    Os 3 últimos são extras usados apenas pelo motor de regras (fallback).
    """

    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    # --- 14 features ML canônicas ---
    especie: Literal["BOVINO", "OVINO", "CAPRINO"]
    raca_femea: str = Field(..., min_length=1, max_length=60)
    condicao_corporal: float = Field(..., ge=1.0, le=5.0,
        description="Escore de condição corporal (1–5, resolução 0.5)")
    historico_taxa_prenhez: float = Field(..., ge=0.0, le=1.0,
        description="Proporção de inseminações anteriores com diagnóstico PRENHA")
    intervalo_pos_parto_dias: int = Field(..., ge=0,
        description="Dias desde o último parto")
    num_partos_anteriores: int = Field(..., ge=0,
        description="Número de partos confirmados do animal")
    dias_desde_ultima_ins: int = Field(..., ge=0,
        description="Dias desde a última inseminação registrada")
    dep_fertilidade_animal: float = Field(...,
        description="DEP de fertilidade do animal (fêmea)")
    tipo_inseminacao: Literal["IATF", "IA_CONVENCIONAL"]
    protocolo_hormonal: str = Field(..., min_length=1, max_length=100)
    temperatura_ambiente_c: float = Field(..., le=50.0,
        description="Temperatura ambiente no momento da inseminação (°C)")
    estacao: Literal["SECA", "CHUVOSA"]
    dep_acuracia: float = Field(..., ge=0.0, le=1.0,
        description="Acurácia média dos DEPs do animal e do reprodutor")
    coeficiente_endogamia: float = Field(..., ge=0.0, le=1.0,
        description="Coeficiente de endogamia F esperado para a prógenie")

    # --- Extras: usados apenas pelo motor de regras ---
    ciclos_sem_concepcao: int = Field(default=0, ge=0,
        description="Ciclos reprodutivos consecutivos sem concepção")
    dep_fertilidade_reprodutor: float | None = Field(default=None,
        description="DEP de fertilidade do reprodutor candidato")
    heterose_esperada: float | None = Field(default=None, ge=0.0,
        description="Heterose esperada do cruzamento (%)")

    @model_validator(mode="after")
    def validar_consistencia(self) -> "PredicaoRequest":
        # IATF deve ter protocolo diferente de IA_CONVENCIONAL
        if self.tipo_inseminacao == "IATF" and self.protocolo_hormonal == "IA_CONVENCIONAL":
            raise ValueError(
                "tipo_inseminacao=IATF requer protocolo_hormonal específico "
                "(não 'IA_CONVENCIONAL')"
            )
        return self


# ---------------------------------------------------------------------------
# POST /predicao — Response
# ---------------------------------------------------------------------------


class FatorDeterminante(BaseModel):
    """Um dos top-5 fatores que mais contribuíram para a predição."""

    feature: str = Field(..., description="Nome técnico da feature")
    valor: Any = Field(..., description="Valor atual da feature para este animal")
    impacto: float = Field(..., description="Contribuição numérica para o score (escala SHAP ou delta/40)")
    sentido: Literal["positivo", "negativo"]
    label: str | None = Field(default=None, description="Descrição humana da condição (motor de regras)")


class PredicaoResponse(BaseModel):
    """Resposta completa do endpoint POST /predicao."""

    predicao_id: str = Field(..., description="UUID único desta predição (gerado pelo microsserviço)")
    score_prenhez: float = Field(..., ge=0.0, le=1.0,
        description="Probabilidade de prenhez no intervalo [0, 1]")
    score_percentual: int = Field(..., ge=0, le=100,
        description="Score arredondado em porcentagem")
    classificacao: Literal["FAVORAVEL", "MEDIO", "DESFAVORAVEL"]
    fatores_determinantes: list[FatorDeterminante] = Field(
        ..., min_length=1, max_length=5,
        description="Top-5 fatores que mais influenciaram o score"
    )
    recomendacoes: list[str] = Field(
        ..., min_length=1, max_length=3,
        description="Recomendações textuais geradas por templates"
    )
    aviso_clinico: str = Field(
        ...,
        description="Aviso obrigatório: predição não substitui julgamento veterinário"
    )
    modelo_versao: str = Field(..., description="Versão do modelo que gerou o resultado")
    motor_utilizado: Literal["ml", "rules"] = Field(
        ..., description="'ml' se Random Forest, 'rules' se fallback determinístico"
    )
    processado_em_ms: int = Field(..., ge=0,
        description="Tempo de processamento no microsserviço em milissegundos")


# ---------------------------------------------------------------------------
# POST /padroes-fertilidade — Request / Response
# ---------------------------------------------------------------------------


class PadroesRequest(BaseModel):
    """
    Dataset enviado pelo backend Java para análise de padrões por K-Means.
    O microsserviço recebe os dados prontos — nunca acessa o banco diretamente.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    inseminacoes: list[dict] = Field(
        ..., min_length=1,
        description="Lista de registros de inseminação com diagnóstico confirmado"
    )
    filtros_aplicados: dict = Field(
        default_factory=dict,
        description="Contexto dos filtros usados pelo backend (para gerar insights)"
    )
    min_clusters: int = Field(default=3, ge=2, le=10)
    max_clusters: int = Field(default=5, ge=2, le=10)

    @model_validator(mode="after")
    def validar_clusters(self) -> "PadroesRequest":
        if self.min_clusters > self.max_clusters:
            raise ValueError("min_clusters deve ser ≤ max_clusters")
        if len(self.inseminacoes) < 20:
            raise ValueError(
                f"Mínimo de 20 inseminações com diagnóstico necessárias "
                f"(recebidas: {len(self.inseminacoes)})"
            )
        return self


class ClusterInsight(BaseModel):
    cluster_id: int
    tamanho: int = Field(..., ge=1)
    perfil: dict[str, float] = Field(..., description="Médias das features do cluster")
    taxa_prenhez: float = Field(..., ge=0.0, le=1.0)
    descricao_textual: str


class PadroesResponse(BaseModel):
    total_inseminacoes_analisadas: int
    clusters: list[ClusterInsight]
    insights_principais: list[str] = Field(..., max_length=5)
    metodologia: dict


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    modelo_versao: str | None = None
    modelo_carregado: bool = False
    timestamp: str


# ---------------------------------------------------------------------------
# GET /model-info — retorna o conteúdo do model_card.json como dict
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Espelho do model_card.json gerado pelo pipeline de treinamento."""

    model_config = ConfigDict(extra="allow")

    modelo_versao: str
    tipo: str
    framework: str
    treinado_em: str
    metricas: dict[str, Any]
    features_ativas: list[str]
