"""Métricas de avaliação para modelos de regressão de MOID.

Centraliza a reversão de transformação log1p e o cálculo de MAE, RMSE e R².
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(y_true, y_pred) -> dict[str, float]:
    """Calcula MAE, RMSE e R² entre os valores reais e preditos na escala linear."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def regression_report(y_true, y_pred_log) -> dict[str, float]:
    """Reverte log1p das predições (np.expm1) e calcula MAE, RMSE e R² contra y_true original."""
    y_pred = np.expm1(y_pred_log)
    return calcular_metricas(y_true, y_pred)


def imprimir_relatorio_metricas(metricas: dict[str, float], titulo: str = "Métricas de Avaliação") -> None:
    """Exibe no terminal um resumo formatado das métricas calculadas."""
    print(f"\n--- {titulo} ---")
    print(f"MAE:  {metricas['mae']:.5f}")
    print(f"RMSE: {metricas['rmse']:.5f}")
    print(f"R²:   {metricas['r2']:.5f}")
