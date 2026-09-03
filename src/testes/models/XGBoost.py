import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


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

# Considerar uso do HistGradientBoostingRegressor para o dataset (dependendo da quantidade de amostras)
# Ele é mais rápido e eficiente para datasets grandes, mas não suporta todas as funcionalidades do GradientBoostingRegressor.

# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor

def make_model_xgb(
        n_jobs=8,
        verbosity=0,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        min_child_weight=10,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_child_weight=min_child_weight,    
            early_stopping_rounds=10, 
            eval_metric="rmse",
            n_jobs=n_jobs,
            # reg_alpha=0.1, estudar se vale a pena usar a regularização L1 ou L2 aqui
            verbosity=verbosity,
            random_state=42
    )

model_xgb = make_model_xgb(verbosity=3)

if __name__ == "__main__":

    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred_log = model_xgb.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Reverter log1p

    mae = mean_absolute_error(y_orig_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred))
    r2 = r2_score(y_orig_test, y_pred)

    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R²: {r2:.5f}")

    importances = dict(zip(X_train.columns, model_xgb.feature_importances_)) # Criar dicionário de importância
    print("\nImportância das features:")

    # '-' para ser decrescente e '[1]' para pegar o valor e não a chave
    for feat, importance in sorted(importances.items(),  key=lambda x: -x[1]):
        print(f"{feat}: {importance:.4f}")

    