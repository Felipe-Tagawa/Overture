import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.config import FEATURES, TARGET
from data.dataset_client import df

X = df[FEATURES]
X = pd.get_dummies(X, columns=['class'], drop_first=True) # RFR não aceita strings
y = df["moid_log"]
y_original = df[TARGET]

X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
    X,
    y,
    y_original,
    test_size=0.2,
    random_state=42
)

def make_model_rf(n_jobs=8, verbose=0, max_depth=20, min_samples_leaf=2, n_estimators=50) -> RandomForestRegressor:

    return RandomForestRegressor(
        n_estimators=n_estimators,
        verbose=verbose,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=42
    )

model_rf = make_model_rf(verbose=2)

if __name__ == "__main__":
    print("Treinando modelo isoladamente no RFR.py...")
    model_rf.fit(X_train, y_train)

    y_pred_log = model_rf.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Reverter log1p

    mae = mean_absolute_error(y_orig_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred))
    r2 = r2_score(y_orig_test, y_pred)

    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R²: {r2:.5f}")

    importances = dict(zip(FEATURES, model_rf.feature_importances_))
    print("\nImportância das features:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")

# Se quiser testar, vai para a raiz: Overture/
# Depois roda python -m model.RFR, tendo como exemplo esse arquivo aqui.
# Obs.1: Cada pasta deve ter um __init__.py para ser entendida como pasta python e poder rodar corretamente.
# Obs.2: Cada import de um outro arquivo deve ser feito da seguinte forma: from pasta-mae.arquivo import arquivo/variável global.
# Seguir essas observacões para não ter problema na hora de rodar.