from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.config import RESULTS_PATH

CSV_PATH = RESULTS_PATH / "autogluon_comparative_models.csv"

COR_SEQ = "#4CA5A8"
COR_PAR = "#F5183D"
CORES = {"sequential": COR_SEQ, "parallel": COR_PAR}
NOMES_ESTRATEGIA = {"sequential": "Sequencial", "parallel": "Parallel"}
ORDEM_ESTRATEGIAS = ["sequential", "parallel"]


def carregar_dados(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Lê o CSV do benchmark e prepara colunas auxiliares."""
    df = pd.read_csv(csv_path)

    # score_val do AutoGluon vem negativo (convenção "maior é melhor");
    # convertendo para RMSE positivo, igual era feito manualmente antes.
    df["score_val_abs"] = df["score_val"].abs()

    return df


def _estrategias_presentes(df: pd.DataFrame) -> list:
    """Mantém a ordem sequencial -> parallel quando ambas existirem no CSV."""
    presentes = df["fit_strategy"].unique().tolist()
    return [e for e in ORDEM_ESTRATEGIAS if e in presentes] + [
        e for e in presentes if e not in ORDEM_ESTRATEGIAS
    ]


def grafico_rmse_por_modelo(df: pd.DataFrame) -> go.Figure:
    """Compara o RMSE de validação de cada modelo, lado a lado por estratégia."""
    fig = go.Figure()

    for estrategia in _estrategias_presentes(df):
        subset = df[df["fit_strategy"] == estrategia]
        fig.add_trace(go.Bar(
            name=NOMES_ESTRATEGIA.get(estrategia, estrategia),
            x=subset["model"],
            y=subset["score_val_abs"],
            marker_color=CORES.get(estrategia),
            text=[f"{v:.5f}" for v in subset["score_val_abs"]],
            textposition="outside",
        ))

    fig.update_layout(
        title="RMSE de validação por modelo",
        xaxis_title="Modelo",
        yaxis_title="RMSE (validação)",
        barmode="group",
        template="plotly_white",
        legend_title="Estratégia",
        height=500,
    )
    return fig


def grafico_tempo_por_modelo(df: pd.DataFrame) -> go.Figure:
    """Compara o tempo de treino de cada modelo, lado a lado por estratégia."""
    fig = go.Figure()

    for estrategia in _estrategias_presentes(df):
        subset = df[df["fit_strategy"] == estrategia]
        fig.add_trace(go.Bar(
            name=NOMES_ESTRATEGIA.get(estrategia, estrategia),
            x=subset["model"],
            y=subset["fit_time"],
            marker_color=CORES.get(estrategia),
            text=[f"{v:.1f}s" for v in subset["fit_time"]],
            textposition="outside",
        ))

    fig.update_layout(
        title="Tempo de treino por modelo",
        xaxis_title="Modelo",
        yaxis_title="Tempo (segundos)",
        barmode="group",
        template="plotly_white",
        legend_title="Estratégia",
        height=500,
    )
    return fig


def grafico_tradeoff_tempo_rmse(df: pd.DataFrame) -> go.Figure:
    """Mostra a relação tempo x RMSE de cada modelo treinado (quanto mais pra baixo e pra esquerda, melhor)."""
    fig = go.Figure()

    for estrategia in _estrategias_presentes(df):
        subset = df[df["fit_strategy"] == estrategia]
        fig.add_trace(go.Scatter(
            name=NOMES_ESTRATEGIA.get(estrategia, estrategia),
            x=subset["fit_time"],
            y=subset["score_val_abs"],
            mode="markers+text",
            text=subset["model"],
            textposition="top center",
            marker=dict(size=14, color=CORES.get(estrategia)),
        ))

    fig.update_layout(
        title="Trade-off entre tempo de treino e RMSE",
        xaxis_title="Tempo de treino (segundos)",
        yaxis_title="RMSE (validação)",
        template="plotly_white",
        legend_title="Estratégia",
        height=500,
    )
    return fig


def grafico_benchmark_final(df: pd.DataFrame) -> go.Figure:
    """Compara o resultado final do pipeline: tempo total e MAE no teste real."""
    estrategias = _estrategias_presentes(df)

    resumo = (
        df[df["fit_strategy"].isin(estrategias)]
        .groupby("fit_strategy")
        .agg(
            tempo=("tempo_total_fit_s", "first"),
            mae=("test_mae_ensemble", "first"),
        )
        .reindex(estrategias)
    )

    nomes = [NOMES_ESTRATEGIA.get(e, e) for e in resumo.index]
    cores = [CORES.get(e, "#999999") for e in resumo.index]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tempo total do pipeline", "MAE no benchmark final"),
    )

    fig.add_trace(go.Bar(
        x=nomes,
        y=resumo["tempo"],
        marker_color=cores,
        text=[f"{v:.2f}s" for v in resumo["tempo"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=nomes,
        y=resumo["mae"],
        marker_color=cores,
        text=[f"{v:.5f}" for v in resumo["mae"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title="Resultado final: Sequencial vs Parallel",
        template="plotly_white",
        height=450,
    )
    return fig


def gerar_html(csv_path: Path = CSV_PATH, caminho_saida: str = "benchmark_overture.html") -> None:
    """Lê o CSV do benchmark e junta os quatro gráficos num único arquivo HTML."""
    df = carregar_dados(csv_path)

    graficos = [
        grafico_benchmark_final(df),
        grafico_rmse_por_modelo(df),
        grafico_tempo_por_modelo(df),
        grafico_tradeoff_tempo_rmse(df),
    ]

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<title>Benchmark Overture - Sequencial vs Parallel</title></head><body>")
        f.write("<h1 style='font-family:sans-serif; text-align:center;'>Benchmark: AutoGluon Sequencial vs Parallel</h1>")

        for i, fig in enumerate(graficos):
            # só carrega o plotly.js na primeira figura, economiza espaço no arquivo
            include_js = "cdn" if i == 0 else False
            f.write(fig.to_html(full_html=False, include_plotlyjs=include_js))

        f.write("</body></html>")

    print(f"HTML gerado em: {caminho_saida}")


if __name__ == "__main__":
    gerar_html()