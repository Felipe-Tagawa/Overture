import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor


from data.config import FEATURES, TARGET
from data.dataset_client import df

LABEL = "moid_log"

data = df[FEATURES + [LABEL, TARGET]]

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

y_orig_test = test_data[TARGET]

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

def make_predictor_autogluon(path: str) -> TabularPredictor:
    return TabularPredictor(
        label=LABEL,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=path,  # diretório próprio por execução, para não sobrescrever
    )


def train_and_evaluate(fit_strategy: str, path: str, num_cpus=16, num_gpus=0, time_limit=900):
    predictor = make_predictor_autogluon(path)

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

    return {
        "fit_strategy": fit_strategy,
        "tempo": elapsed,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictor": predictor,
    }


if __name__ == "__main__":

    resultado_sequencial = train_and_evaluate(
        fit_strategy="sequential",
        path="AutogluonModels/moid_sequential",
    )

    resultado_paralelo = train_and_evaluate(
        fit_strategy="parallel",
        path="AutogluonModels/moid_parallel",
    )

    print("\n=== Comparação sequential vs. parallel (fit_strategy) ===")
    for r in (resultado_sequencial, resultado_paralelo):
        print(
            f"\n[{r['fit_strategy']}] "
            f"Tempo: {r['tempo']:.2f}s | "
            f"MAE: {r['mae']:.5f} | RMSE: {r['rmse']:.5f} | R²: {r['r2']:.5f}"
        )

    diff = resultado_sequencial["tempo"] - resultado_paralelo["tempo"]
    print(f"\nDiferença de tempo (sequential - parallel): {diff:.2f}s")

    print("\nLeaderboard (sequential):")
    print(resultado_sequencial["predictor"].leaderboard(test_data[FEATURES + [LABEL]], silent=True))

    print("\nLeaderboard (parallel):")
    print(resultado_paralelo["predictor"].leaderboard(test_data[FEATURES + [LABEL]], silent=True))