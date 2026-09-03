import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


from src.data.config import FEATURES, TARGET
from src.data.dataset_client import df

X = df[FEATURES]
y = df["moid_log"]
y_original = df[TARGET]

X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
    X,
    y,
    y_original,
    test_size=0.2,
    random_state=42
)

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

# Anotação das Diferenças em relação ao XGBRegressor:
# - Não existe "eval_set" no fit(): o early stopping usa uma fração interna
#   do próprio conjunto de treino (validation_fraction), controlada por early_stopping + n_iter_no_change.
# - Não existe "subsample" (sem amostragem de linhas) nem "min_child_weight".
#   O controle de complexidade da árvore é feito via max_leaf_nodes, min_samples_leaf e l2_regularization.
# - Não existe feature_importances_ nativo (é um modelo baseado em histogramas, não guarda ganho por split).
#  Importância é calculada via permutation_importance depois do treino.

def make_model_hgb(
        n_iter_no_change=10,
        verbose=0,
        max_iter=300, # equivalente a n_estimators no XGBoost (obs: numero de 300 utilizado devido ao early stopping)   
        learning_rate=0.1,
        max_depth=6,
        max_leaf_nodes=31, # controle de complexidade principal do HGB (default sklearn)
        min_samples_leaf=20, # equivalente conceitual ao min_child_weight
        l2_regularization=0.0,
        validation_fraction=0.2, # fração do treino usada internamente para early stopping
) -> HistGradientBoostingRegressor:
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
        scoring="loss", # métrica usada internamente para o early stopping
        verbose=verbose,
        random_state=42
    )

model_hgb = make_model_hgb(verbose=1)

if __name__ == "__main__":

    # Sem eval_set/verbose no fit: o early stopping é resolvido internamente
    # a partir de validation_fraction + n_iter_no_change definidos no modelo.
    model_hgb.fit(X_train, y_train)

    y_pred_log = model_hgb.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # Reverter log1p

    mae = mean_absolute_error(y_orig_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred))
    r2 = r2_score(y_orig_test, y_pred)

    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R²: {r2:.5f}")

    # lembrete: HistGradientBoostingRegressor não possui feature_importances_.
    # Usamos permutation_importance sobre o conjunto de teste como
    # alternativa (mede a queda de performance ao embaralhar cada feature).
    perm_result = permutation_importance(
        model_hgb,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2",
        n_jobs=8,
    )

    importances = dict(zip(X_train.columns, perm_result.importances_mean))
    print("\nImportância das features (permutation importance):")

    for feat, importance in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"{feat}: {importance:.4f}")