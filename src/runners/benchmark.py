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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor


from src.data.config import FEATURES, TARGET, RESULTS_PATH

LABEL = "moid_log"

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
        time_limit: int = 100,
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

    y_pred_log = predictor.predict(test_data[FEATURES])
    y_pred = np.expm1(y_pred_log)  # Reverter log1p

    mae = mean_absolute_error(y_orig_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred))
    r2 = r2_score(y_orig_test, y_pred)

    # Extrai o leaderboard detalhado no conjunto de teste
    df_leaderboard = predictor.leaderboard(test_data[FEATURES + [LABEL]], silent=True)

    # Insere as colunas de contexto da execução
    df_leaderboard["fit_strategy"] = fit_strategy
    df_leaderboard["tempo_total_fit_s"] = elapsed
    df_leaderboard["test_mae_ensemble"] = mae
    df_leaderboard["test_rmse_ensemble"] = rmse
    df_leaderboard["test_r2_ensemble"] = r2

    return df_leaderboard

def print_detailed_models(df_result: pd.DataFrame, strategy: str):
    subset = df_result[df_result["fit_strategy"] == strategy]
    
    print(f"\n Modelos Treinados - Estratégia: {strategy.upper()}")
    print("-" * 65)
    print(f"{'Modelo':<25} | {'Val Score (RMSE)':<18} | {'Tempo Fit (s)':<12}")
    print("-" * 65)
    
    for _, row in subset.iterrows():
        model_name = row["model"]
        val_score = abs(row["score_val"])  # Converte RMSE negativo para positivo
        fit_time = row["fit_time"]
        print(f"{model_name:<25} | {val_score:<18.5f} | {fit_time:<12.2f}")
    print("-" * 65)

def run_benchmark(df: pd.DataFrame) -> pd.DataFrame:

    sequential_result = train_and_evaluate(
        df = df,
        fit_strategy="sequential",
        model_path="AutogluonModels/moid_sequential",
    )

    paralell_result = train_and_evaluate(
        df=df,
        fit_strategy="parallel",
        model_path="AutogluonModels/moid_parallel",
    )

    df_result = pd.concat([sequential_result, paralell_result], ignore_index=True)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    csv_out = RESULTS_PATH / "autogluon_comparative_models.csv"
    parquet_out = RESULTS_PATH / "autogluon_comparative_models.parquet"

    # Exporta os resultados para CSV e Parquet
    df_result.to_csv(csv_out, index=False)
    df_result.to_parquet(parquet_out, index=False)
    print(f"Arquivos salvos em: {RESULTS_PATH}")

    print_detailed_models(df_result, "sequential")
    print_detailed_models(df_result, "parallel")

    t_seq = sequential_result["tempo_total_fit_s"].iloc[0]
    t_par = paralell_result["tempo_total_fit_s"].iloc[0]

    print("\n" + "-" *50)
    print("BENCHMARK")
    print("=" * 50)

    print(f"Sequencial | Tempo: {t_seq:.2f}s | MAE: {sequential_result['test_mae_ensemble'].iloc[0]:.5f}")
    print(f"Parallel   | Tempo: {t_par:.2f}s | MAE: {paralell_result['test_mae_ensemble'].iloc[0]:.5f}")
    print(f"Diferença de tempo (seq - par): {t_seq - t_par:.2f}s")
    print("=" * 50)

    return df_result