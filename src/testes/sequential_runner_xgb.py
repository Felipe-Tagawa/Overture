import os
import sysconfig
import time
from src.testes.models.XGBoost import X_train, X_test, y_train, y_test, make_model_xgb

supports_no_gil = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def train_model(model, name):
    start = time.perf_counter()
    # eval_set obrigatório por causa do early_stopping_rounds configurado
    # em make_model_xgb(). verbose=False para manter o log limpo, já que
    # o objetivo aqui é comparar tempo total, não acompanhar cada árvore.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    end = time.perf_counter()
    return f"Fim {name} em {end - start:.2f}s (best_iteration={model.best_iteration})"


if __name__ == "__main__":
    print(f"Free-threaded: {supports_no_gil}")

    n_models = 4

    # Sequencial: só um modelo treina por vez, então cada um pode
    # usar todos os núcleos internamente sem competir com outro.
    jobs = [
        (make_model_xgb(n_jobs=16, verbosity=0), f"XGBoost_{i+1}") for i in range(n_models)
    ]

    print(f"Modelos: {n_models} (sequencial)")

    initial_time = time.perf_counter()
    result = [train_model(model, name) for model, name in jobs]
    total = time.perf_counter() - initial_time

    print("\nResultados:")
    for r in result:
        print(f"  {r}")
    print(f"\nTempo Total sequencial ({n_models} modelos): {total:.2f}s")