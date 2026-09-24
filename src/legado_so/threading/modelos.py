"""Fábricas de instâncias de modelos para os experimentos de concorrência.

Reúne make_model_rf, make_model_xgb e make_model_hgb de forma desacoplada,
sem disparar carregamento de dados no momento da importação.
"""

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def make_model_rf(
        n_jobs: int = 8,
        verbose: int = 0,
        max_depth: int = 20,
        min_samples_leaf: int = 2,
        n_estimators: int = 50,
        random_state: int = 42,
) -> RandomForestRegressor:
    """Cria uma instância configurada de RandomForestRegressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        verbose=verbose,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def make_model_xgb(
        n_jobs: int = 8,
        verbosity: int = 0,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        min_child_weight: int = 10,
        early_stopping_rounds: int = 10,
        random_state: int = 42,
):
    """Cria uma instância configurada de XGBRegressor.

    A importação do xgboost é tardia para permitir uso em ambientes
    onde a dependência é opcional.
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_child_weight=min_child_weight,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        n_jobs=n_jobs,
        verbosity=verbosity,
        random_state=random_state,
    )


def make_model_hgb(
        n_iter_no_change: int = 10,
        verbose: int = 0,
        max_iter: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        validation_fraction: float = 0.2,
        random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """Cria uma instância configurada de HistGradientBoostingRegressor."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        scoring="loss",
        verbose=verbose,
        random_state=random_state,
    )
