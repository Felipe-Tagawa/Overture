import sysconfig
import time
from joblib import Parallel, delayed, parallel_config
from testes.models.XGBoost import X_train, X_test, y_train, y_test, make_model_xgb

supports_no_gil = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def train_model(model, name):
    start = time.perf_counter()
    # XGBoost, diferente do RandomForest, precisa de eval_set no fit()
    # por causa do early_stopping_rounds definido em make_model_xgb().
    # verbose=False aqui para não misturar o log de cada árvore com os
    # outros modelos rodando em paralelo.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    end = time.perf_counter()
    return f"Fim {name} em {end - start:.2f}s (best_iteration={model.best_iteration})"

if __name__ == "__main__":

    print(f"Free-threaded: {supports_no_gil}")

    n_models = 4  # Modelos para treinar em paralelo
    n_outer_jobs = min(n_models, 16)
    n_inner_jobs = max(1, 16 // n_outer_jobs)

    # verbosity=0 para não poluir o log com o treino de cada árvore
    # em paralelo (diferente do model_xgb.py, que roda isolado).
    jobs = [
        (make_model_xgb(n_jobs=n_inner_jobs, verbosity=0), f"XGBoost_{i+1}") for i in range(n_models)
    ]

    print(f"Modelos em paralelo: {n_outer_jobs} | Threads internas por modelo: {n_inner_jobs}")

    with parallel_config(backend="threading", n_jobs=n_outer_jobs):
        initial_time = time.perf_counter()

        result = Parallel()(
            delayed(train_model)(model, name) for model, name in jobs
        )

        total = time.perf_counter() - initial_time

    print("\nResultados:")
    for r in result:
        print(f"  {r}")
    print(f"\nTempo Total ({n_outer_jobs} threads externas x {n_inner_jobs} internas): {total:.2f}s")