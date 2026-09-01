import os
import sysconfig
import time
from models.RFR import X, y, make_model_rf

supports_no_gil = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def train_model(model, name):
    start = time.perf_counter()
    model.fit(X, y)
    end = time.perf_counter()
    return f"Fim {name} em {end - start:.2f}s"


if __name__ == "__main__":
    print(f"Free-threaded: {supports_no_gil}")

    n_models = 4

    # Sequencial: só um modelo treina por vez, então cada um pode
    # usar todos os núcleos internamente sem competir com outro.
    jobs = [
        (make_model_rf(n_jobs=16, verbose=2), f"RandomForest_{i+1}") for i in range(n_models)
    ]

    print(f"Modelos: {n_models} (sequencial)")

    initial_time = time.perf_counter()
    result = [train_model(model, name) for model, name in jobs]
    total = time.perf_counter() - initial_time

    print("\nResultados:")
    for r in result:
        print(f"  {r}")
    print(f"\nTempo Total sequencial ({n_models} modelos): {total:.2f}s")