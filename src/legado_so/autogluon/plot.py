from pathlib import Path
from typing import Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.comum.config import RESULTS_PATH

CSV_PATH = RESULTS_PATH / "autogluon_comparative_models.csv"
HTML_PATH = RESULTS_PATH / "benchmark_overture.html"

COR_SEQ = "#4CA863"
COR_PAR = "#F56918"
CORES = {"sequential": COR_SEQ, "parallel": COR_PAR}
NOMES_ESTRATEGIA = {"sequential": "Sequencial", "parallel": "Parallel"}
ORDEM_ESTRATEGIAS = ["sequential", "parallel"]

# Informações utilizadas para gerar a tabela de leaderboard:
# Cores usadas só na tabela (independentes da paleta sequential/parallel acima,
# aqui as cores marcam "linha destaque" vs "linha normal", não estratégia)
COR_HEADER_TABELA = "#1f2937"
COR_LINHA_PAR = "#f3f4f6"
COR_LINHA_IMPAR = "#ffffff"
COR_MELHOR_LINHA = "#d1fae5"

COLUNAS_TABELA = ["model", "score_test", "score_val", "pred_time_test", "fit_time_marginal", "fit_order"]
LABELS_TABELA = ["Modelo", "Score (test)", "Score (val)", "Pred. time (s)", "Fit time (s)", "Ordem"]


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
        title="Resultado final: Sequencial vs Paralelo",
        template="plotly_white",
        height=450,
    )
    return fig


def grafico_tabela_leaderboard(df: pd.DataFrame) -> go.Figure:
    """Gera uma tabela por estratégia (sequential/parallel), lado a lado,
    destacando o melhor modelo (menor score_test) de cada uma."""

    estrategias = _estrategias_presentes(df)

    fig = make_subplots(
        rows=1, cols=len(estrategias),
        specs=[[{"type": "table"}] * len(estrategias)],
        subplot_titles=[f"Leaderboard — {NOMES_ESTRATEGIA.get(e, e)}" for e in estrategias],
        horizontal_spacing=0.03,
    )

    max_linhas = 0

    for col_idx, estrategia in enumerate(estrategias, start=1):
        subset = df[df["fit_strategy"] == estrategia].copy()
        subset["score_test_abs"] = subset["score_test"].abs()
        subset = subset.sort_values("score_test_abs").reset_index(drop=True)
        max_linhas = max(max_linhas, len(subset))

        melhor_pos = subset["score_test_abs"].idxmin()

        cores_linha = [
            COR_MELHOR_LINHA if pos == melhor_pos
            else (COR_LINHA_PAR if pos % 2 == 0 else COR_LINHA_IMPAR)
            for pos in subset.index
        ]

        valores = [
            subset["model"].tolist(),
            [f"{v:.5f}" for v in subset["score_test"]],
            [f"{v:.5f}" for v in subset["score_val"]],
            [f"{v:.3f}" for v in subset["pred_time_test"]],
            [f"{v:.2f}" for v in subset["fit_time_marginal"]],
            subset["fit_order"].astype(int).astype(str).tolist(),
        ]

        fig.add_trace(go.Table(
            header=dict(
                values=LABELS_TABELA,
                fill_color=COR_HEADER_TABELA,
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=valores,
                fill_color=[cores_linha] * len(COLUNAS_TABELA),
                align="center",
                height=28,
                font=dict(size=11),
            ),
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Comparação detalhada dos modelos (AutoGluon)",
        template="plotly_white",
        height=max(300, 190 + 32 * (max_linhas + 1)),
        width=900 + 900 * (len(estrategias) - 1),
    )
    return fig


def grafico_evolucao_time_limit(df: pd.DataFrame) -> go.Figure:
    """Mostra como o tempo total e o MAE do ensemble evoluem conforme o time_limit aumenta,
    uma linha por estratégia. É o gráfico que responde 'vale a pena dar mais tempo?'."""

    resumo = (
        df.groupby(["fit_strategy", "time_limit"])
        .agg(
            tempo=("tempo_total_fit_s", "first"),
            mae=("test_mae_ensemble", "first"),
        )
        .reset_index()
        .sort_values("time_limit")
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tempo total x Time Limit", "MAE do ensemble x Time Limit"),
    )

    for estrategia in _estrategias_presentes(df):
        sub = resumo[resumo["fit_strategy"] == estrategia]
        nome = NOMES_ESTRATEGIA.get(estrategia, estrategia)
        cor = CORES.get(estrategia)

        fig.add_trace(go.Scatter(
            x=sub["time_limit"], y=sub["tempo"],
            mode="lines+markers+text",
            text=[f"{v:.1f}s" for v in sub["tempo"]],
            textposition="top center",
            name=nome, legendgroup=estrategia,
            marker=dict(color=cor, size=10),
            line=dict(color=cor),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sub["time_limit"], y=sub["mae"],
            mode="lines+markers+text",
            text=[f"{v:.5f}" for v in sub["mae"]],
            textposition="top center",
            name=nome, legendgroup=estrategia, showlegend=False,
            marker=dict(color=cor, size=10),
            line=dict(color=cor),
        ), row=1, col=2)

    fig.update_xaxes(title_text="Time limit (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time limit (s)", row=1, col=2)
    fig.update_yaxes(title_text="Tempo total (s)", row=1, col=1)
    fig.update_yaxes(title_text="MAE (teste)", row=1, col=2)

    fig.update_layout(
        title="Evolução por Time Limit: Sequencial vs Paralelo",
        template="plotly_white",
        height=450,
    )
    return fig


def gerar_html(csv_path: Path = CSV_PATH, caminho_saida: Union[Path, str] = HTML_PATH) -> None:
    """Lê o CSV do benchmark e monta o HTML gravando em resultados/: um resumo geral por time_limit,
    seguido do detalhamento (RMSE, tempo, trade-off, leaderboard) para cada time_limit."""
    df = carregar_dados(csv_path)

    time_limits = sorted(df["time_limit"].unique()) if "time_limit" in df.columns else [None]

    saida = Path(caminho_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    with open(saida, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<title>Benchmark Overture - Sequencial vs Paralelo</title></head><body>")
        f.write("<h1 style='font-family:sans-serif; text-align:center;'>Benchmark: AutoGluon Sequencial vs Paralelo</h1>")

        primeira_figura = True

        if len(time_limits) > 1:
            fig_evolucao = grafico_evolucao_time_limit(df)
            f.write(fig_evolucao.to_html(full_html=False, include_plotlyjs="cdn"))
            primeira_figura = False

        for tl in time_limits:
            subset = df[df["time_limit"] == tl] if tl is not None else df

            if tl is not None:
                f.write(f"<h2 style='font-family:sans-serif; text-align:center; margin-top:60px;'>Time limit: {tl}s</h2>")

            graficos = [
                grafico_benchmark_final(subset),
                grafico_rmse_por_modelo(subset),
                grafico_tempo_por_modelo(subset),
                grafico_tradeoff_tempo_rmse(subset),
                grafico_tabela_leaderboard(subset),
            ]

            for fig in graficos:
                include_js = "cdn" if primeira_figura else False
                f.write(fig.to_html(full_html=False, include_plotlyjs=include_js))
                primeira_figura = False

        f.write("</body></html>")

    print(f"HTML gerado em: {saida}")


if __name__ == "__main__":
    gerar_html()
