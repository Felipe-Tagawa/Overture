# https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
#
# O AutoGluon já possui um parâmetro nativo para controlar o paralelismo
# do treino do portfólio de modelos: fit_strategy.
#   - "sequential" (padrão): cada modelo do portfólio treina um de cada
#     vez, usando todos os CPUs liberados para ele.
#   - "parallel": os modelos treinam simultaneamente via Ray, dividindo
#     os CPUs disponíveis entre eles. Experimental desde a versão 1.2.0,
#     e ainda sem suporte a GPU.
#
# Isso é o equivalente, dentro do AutoGluon, ao que foi feito manualmente
# com joblib(backend="threading") nos runners de RFR/XGBoost — só que
# aqui é o próprio framework quem decide como distribuir o trabalho entre
# os modelos do portfólio, usando Ray como motor de paralelismo.
#
# IMPORTANTE: cada .fit() com fit_strategy="parallel" (assim como bagging
# e dynamic stacking) inicializa sua própria sessão do Ray internamente.
# A documentação do AutoGluon recomenda não inicializar múltiplos runtimes
# do Ray no mesmo processo — por isso as duas execuções abaixo rodam uma
# após a outra (não em threads Python simultâneas). O paralelismo que
# queremos medir já acontece DENTRO de cada chamada de fit(), não entre
# chamadas.
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

from src.comum.config import FEATURES, TARGET, LABEL, RESULTS_PATH
from src.comum.metricas import regression_report


def make_predictor_autogluon(path: str) -> TabularPredictor:
    return TabularPredictor(
        label=LABEL,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=path,  # diretório próprio por execução, para não sobrescrever
    )


def train_and_evaluate(
        df: pd.DataFrame,
        fit_strategy: str,
        model_path: str,
        num_cpus: int = 16,
        num_gpus: int = 0,
        time_limit: int = 50,
) -> pd.DataFrame:

    data = df[FEATURES + [LABEL, TARGET]]

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42
    )

    y_orig_test = test_data[TARGET]

    predictor = make_predictor_autogluon(model_path)

    start = time.perf_counter()
    predictor.fit(
        train_data[FEATURES + [LABEL]],
        presets="medium_quality",
        time_limit=time_limit,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        fit_strategy=fit_strategy,
    )
    elapsed = time.perf_counter() - start

    # Extrai o leaderboard detalhado no conjunto de teste
    df_leaderboard = predictor.leaderboard(test_data[FEATURES + [LABEL]], silent=True)

    mae_per_model = []
    rmse_per_model = []
    r2_per_model = []

    for model_name in df_leaderboard["model"]:
        y_pred_log = predictor.predict(test_data[FEATURES + [LABEL]], model=model_name)
        metrics = regression_report(y_orig_test, y_pred_log)

        mae_per_model.append(metrics["mae"])
        rmse_per_model.append(metrics["rmse"])
        r2_per_model.append(metrics["r2"])

    df_leaderboard["test_mae"] = mae_per_model
    df_leaderboard["test_rmse"] = rmse_per_model
    df_leaderboard["test_r2"] = r2_per_model

    df_leaderboard["fit_strategy"] = fit_strategy
    df_leaderboard["tempo_total_fit_s"] = elapsed
    df_leaderboard["test_mae_ensemble"] = df_leaderboard.loc[
        df_leaderboard["model"] == "WeightedEnsemble_L2", "test_mae"
    ].iloc[0]
    df_leaderboard["test_rmse_ensemble"] = df_leaderboard.loc[
        df_leaderboard["model"] == "WeightedEnsemble_L2", "test_rmse"
    ].iloc[0]
    df_leaderboard["test_r2_ensemble"] = df_leaderboard.loc[
        df_leaderboard["model"] == "WeightedEnsemble_L2", "test_r2"
    ].iloc[0]

    return df_leaderboard


def print_detailed_models(df_result: pd.DataFrame, strategy: str, time_limit: int = None):
    subset = df_result[df_result["fit_strategy"] == strategy]
    if time_limit is not None:
        subset = subset[subset["time_limit"] == time_limit]

    tempo_total = subset["tempo_total_fit_s"].iloc[0]

    header = f"\n Modelos Treinados - Estratégia: {strategy.upper()}"
    if time_limit is not None:
        header += f" | Time Limit: {time_limit}s"
    print(header)
    print("-" * 100)
    print(f"{'Modelo':<25} | {'Val Score (RMSE)':<18} | {'Tempo Acumulado (s)':<20}")
    print("-" * 100)

    for _, row in subset.iterrows():
        model_name = row["model"]
        val_score = abs(row["score_val"])
        fit_cumulative = row["fit_time"]

        print(f"{model_name:<25} | {val_score:<18.5f} | {fit_cumulative:<20.2f}")

    print("-" * 100)
    print(f"Tempo total da estratégia: {tempo_total:.2f}s")


def print_benchmark_summary(df_result: pd.DataFrame, time_limits: list[int]):
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARATIVO POR TIME LIMIT")
    print("=" * 70)
    print(f"{'Time Limit':<12} | {'Estratégia':<12} | {'Tempo Real (s)':<15} | {'MAE':<10}")
    print("-" * 70)

    for tl in time_limits:
        for strategy in ["sequential", "parallel"]:
            row = df_result[
                (df_result["time_limit"] == tl) & (df_result["fit_strategy"] == strategy)
            ].iloc[0]
            print(f"{tl:<12} | {strategy:<12} | {row['tempo_total_fit_s']:<15.2f} | {row['test_mae_ensemble']:<10.5f}")
    print("=" * 70)


def run_benchmark(df: pd.DataFrame, time_limits: list[int] = [25, 50, 100]) -> pd.DataFrame:
    all_results = []

    for tl in time_limits:
        sequential_result = train_and_evaluate(
            df=df,
            fit_strategy="sequential",
            model_path=f"AutogluonModels/moid_sequential_{tl}s",
            time_limit=tl,
        )
        sequential_result["time_limit"] = tl

        parallel_result = train_and_evaluate(
            df=df,
            fit_strategy="parallel",
            model_path=f"AutogluonModels/moid_parallel_{tl}s",
            time_limit=tl,
        )
        parallel_result["time_limit"] = tl

        all_results.append(sequential_result)
        all_results.append(parallel_result)

        print_detailed_models(sequential_result, "sequential", tl)
        print_detailed_models(parallel_result, "parallel", tl)

    df_result = pd.concat(all_results, ignore_index=True)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_PATH / "autogluon_comparative_models.csv"
    parquet_out = RESULTS_PATH / "autogluon_comparative_models.parquet"
    df_result.to_csv(csv_out, index=False)
    df_result.to_parquet(parquet_out, index=False)
    print(f"\nArquivos salvos em: {RESULTS_PATH}")

    print_benchmark_summary(df_result, time_limits)

    return df_result
