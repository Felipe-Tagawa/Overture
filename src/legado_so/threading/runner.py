"""Runner unificado para experimentos de concorrência (sequencial vs threads)
com modelos Random Forest, XGBoost e HistGradientBoosting.

Uso via CLI:
    python -m src.legado_so.threading.runner --modelo {rf,xgb,hgb} --estrategia {seq,thread} [--n_models N] [--max_cpus C]
"""

import argparse
import sysconfig
import time
from sklearn.model_selection import train_test_split

from src.comum.config import FEATURES, LABEL, TARGET
from src.comum.dados import get_clean_data
from src.legado_so.executar import aplicar_escopo_legado
from src.legado_so.threading.modelos import (
    make_model_rf,
    make_model_xgb,
    make_model_hgb,
)

supports_no_gil = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def carregar_dados_treino_teste():
    """Carrega os dados com o escopo legado e divide em treino e teste."""
    df_clean = get_clean_data(force_reprocess=False)
    df_escopo = aplicar_escopo_legado(df_clean)

    X = df_escopo[FEATURES]
    y = df_escopo[LABEL]
    y_orig = df_escopo[TARGET]

    return train_test_split(X, y, y_orig, test_size=0.2, random_state=42)


def treinar_modelo(modelo, nome, X_train, y_train, X_test=None, y_test=None):
    """Treina um modelo individual e mede o tempo de execução."""
    start = time.perf_counter()

    # Modelos com early stopping (como XGBoost) necessitam de eval_set
    if hasattr(modelo, "early_stopping_rounds") and modelo.early_stopping_rounds and X_test is not None:
        modelo.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        end = time.perf_counter()
        best_iter = getattr(modelo, "best_iteration", None)
        iter_info = f" (best_iteration={best_iter})" if best_iter is not None else ""
        return f"Fim {nome} em {end - start:.2f}s{iter_info}"
    else:
        modelo.fit(X_train, y_train)
        end = time.perf_counter()
        return f"Fim {nome} em {end - start:.2f}s"


def criar_jobs(modelo_tipo: str, n_models: int, n_jobs_por_modelo: int):
    """Cria a lista de instâncias de modelo e nomes baseada no tipo solicitado."""
    jobs = []
    for i in range(n_models):
        nome = f"{modelo_tipo.upper()}_{i + 1}"
        if modelo_tipo == "rf":
            m = make_model_rf(n_jobs=n_jobs_por_modelo, verbose=0)
        elif modelo_tipo == "xgb":
            m = make_model_xgb(n_jobs=n_jobs_por_modelo, verbosity=0)
        elif modelo_tipo == "hgb":
            m = make_model_hgb(verbose=0)
        else:
            raise ValueError(f"Modelo desconhecido: '{modelo_tipo}'. Opções: 'rf', 'xgb', 'hgb'.")
        jobs.append((m, nome))
    return jobs


def executar_runner(modelo_tipo: str = "rf", estrategia: str = "seq", n_models: int = 4, max_cpus: int = 16):
    """Executa o experimento de treino conforme os parâmetros definidos."""
    print(f"Free-threaded (no-GIL): {supports_no_gil}")

    X_train, X_test, y_train, y_test, _, _ = carregar_dados_treino_teste()

    if estrategia == "seq":
        print(f"Modelo: {modelo_tipo.upper()} | Estratégia: Sequencial | Modelos: {n_models}")
        jobs = criar_jobs(modelo_tipo, n_models, n_jobs_por_modelo=max_cpus)

        initial_time = time.perf_counter()
        resultados = [
            treinar_modelo(m, nome, X_train, y_train, X_test, y_test)
            for m, nome in jobs
        ]
        total = time.perf_counter() - initial_time

        print("\nResultados:")
        for r in resultados:
            print(f"  {r}")
        print(f"\nTempo Total sequencial ({n_models} modelos): {total:.2f}s")

    elif estrategia == "thread":
        from joblib import Parallel, delayed, parallel_config

        n_outer_jobs = min(n_models, max_cpus)
        n_inner_jobs = max(1, max_cpus // n_outer_jobs)
        print(
            f"Modelo: {modelo_tipo.upper()} | Estratégia: Threads | "
            f"Modelos em paralelo: {n_outer_jobs} | Threads internas por modelo: {n_inner_jobs}"
        )

        jobs = criar_jobs(modelo_tipo, n_models, n_jobs_por_modelo=n_inner_jobs)

        with parallel_config(backend="threading", n_jobs=n_outer_jobs):
            initial_time = time.perf_counter()
            resultados = Parallel()(
                delayed(treinar_modelo)(m, nome, X_train, y_train, X_test, y_test)
                for m, nome in jobs
            )
            total = time.perf_counter() - initial_time

        print("\nResultados:")
        for r in resultados:
            print(f"  {r}")
        print(
            f"\nTempo Total ({n_outer_jobs} threads externas x {n_inner_jobs} internas): {total:.2f}s"
        )
    else:
        raise ValueError(f"Estratégia inválida: '{estrategia}'. Opções: 'seq' ou 'thread'.")


def main():
    parser = argparse.ArgumentParser(description="Runner de concorrência para modelos de regressão de MOID.")
    parser.add_argument("--modelo", choices=["rf", "xgb", "hgb"], default="rf", help="Modelo a treinar (rf, xgb, hgb)")
    parser.add_argument("--estrategia", choices=["seq", "thread"], default="seq", help="Estratégia de concorrência (seq, thread)")
    parser.add_argument("--n_models", type=int, default=4, help="Número de modelos a treinar (padrão: 4)")
    parser.add_argument("--max_cpus", type=int, default=16, help="CPUs lógicas disponíveis (padrão: 16)")

    args = parser.parse_args()
    executar_runner(
        modelo_tipo=args.modelo,
        estrategia=args.estrategia,
        n_models=args.n_models,
        max_cpus=args.max_cpus,
    )


if __name__ == "__main__":
    main()
