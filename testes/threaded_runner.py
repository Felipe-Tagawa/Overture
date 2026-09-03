import sysconfig
import time
from joblib import Parallel, delayed, parallel_config
from testes.models.RFR import X, y, make_model_rf

supports_no_gil = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def train_model(model, name):
    start = time.perf_counter()
    model.fit(X, y)
    end = time.perf_counter()
    return f"Fim {name} em {end - start:.2f}s"

if __name__ == "__main__":

    print(f"Free-threaded: {supports_no_gil}")

    n_models = 4 # Modelos para treinar em paralelo
    n_outer_jobs = min(n_models, 16)
    n_inner_jobs = max(1, 16 // n_outer_jobs)

    jobs = [
    (make_model_rf(n_jobs=n_inner_jobs), f"RandomForest_{i+1}") for i in range(n_models)
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