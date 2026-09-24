"""Ponto de entrada para a frente de Sistemas Operacionais (congelada).

Aplica os filtros de escopo originais (a < 5, e < 0.9), calcula moid_log
e executa o benchmark comparativo com AutoGluon (sequencial vs paralelo).

Execução:
    python -m src.legado_so.executar
"""

import numpy as np
import pandas as pd

from src.comum.config import TARGET, LABEL
from src.comum.dados import get_clean_data
from src.legado_so.autogluon.benchmark import run_benchmark


def aplicar_escopo_legado(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o filtro congelado da frente de SO (a < 5, e < 0.9) e calcula moid_log."""
    df_escopo = df[(df["a"] < 5) & (df["e"] < 0.9)].copy()

    if TARGET in df_escopo.columns:
        df_escopo[LABEL] = np.log1p(df_escopo[TARGET])

    return df_escopo


def main():
    df_clean = get_clean_data(force_reprocess=False)
    df_final = aplicar_escopo_legado(df_clean)

    print(f"Total de Registros: {len(df_final)}")
    print(f"Colunas Disponíveis: {list(df_final.columns)}")

    run_benchmark(df=df_final)


if __name__ == "__main__":
    main()
