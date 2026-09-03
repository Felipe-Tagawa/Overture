import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor


from src.data.config import FEATURES, TARGET
from src.data.dataset_client import df

LABEL = "moid_log" #nome da clouna alvo

# Diferente dos modelos nossos tipo XGBoost/HistGB/RFR, o AutoGluon trabalha com o DataFrame
# completo (features + label juntos), não com as features e targets em X e y separados. Por isso
# mantem aqui as colunas de FEATURES + o label + o TARGET original (esse último só para conseguir reverter o log dps do split).
data = df[FEATURES + [LABEL, TARGET]]

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

y_orig_test = test_data[TARGET]

# https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html
# https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html

def make_predictor_autogluon(
        label=LABEL,
        eval_metric="root_mean_squared_error",
        problem_type="regression",
) -> TabularPredictor:
    return TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        # path="AutogluonModels/moid_predictor",  # opcional: é pra fixar diretório de saída
    )

predictor_autogluon = make_predictor_autogluon()

if __name__ == "__main__":

    # Diferente do .fit() dos outros modelos, aqui não passamos X/y nem
    # eval_set: o AutoGluon recebe o DataFrame de treino completo e faz
    # sozinho o split interno de validação (holdout) para HPO/bagging.
    
    # num_cpus / num_gpus / presets / time_limit controlam o grau deparalelismo e o orçamento de tempo do 
    # treino do portfólio de modelos, são os parâmetros centrais para o experimento de threads do grupo.
    predictor_autogluon.fit(
        train_data[FEATURES + [LABEL]],
        presets="medium_quality",
        time_limit=600,# segundos: ajustar conforme o orçamento do experimento
        num_cpus=16,
        num_gpus=0,
    )

    y_pred_log = predictor_autogluon.predict(test_data[FEATURES])
    y_pred = np.expm1(y_pred_log)  # Reverter log1p

    mae = mean_absolute_error(y_orig_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_orig_test, y_pred))
    r2 = r2_score(y_orig_test, y_pred)

    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R²: {r2:.5f}")

    # Leaderboard: ranking de todos os modelos do portfólio treinados
    # internamente pelo AutoGluon (não existe equivalente nos scripts anteriores, pois eles treinam um único modelo).
    print("\nLeaderboard dos modelos treinados:")
    print(predictor_autogluon.leaderboard(test_data[FEATURES + [LABEL]], silent=True))

    # feature_importance nativo do AutoGluon já é baseado em permutation
    # importance (mesmo princípio usado no script do HistGradientBoosting),calculado sobre o modelo final (ensemble).
    importances = predictor_autogluon.feature_importance(test_data[FEATURES + [LABEL]])
    print("\nImportância das features (permutation importance):")
    print(importances)