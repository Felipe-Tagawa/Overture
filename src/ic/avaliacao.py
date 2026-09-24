"""Módulo de Avaliação e Baselines Físicos.

1. BASELINE ANALÍTICO FÍSICO-GEOMÉTRICO: max(q - 1, 0)
   - Motivação científica:
     Antes de treinar redes neurais ou ensembles pesados (como AutoGluon, XGBoost, etc.),
     é essencial responder: "um modelo de ML realmente aprende a física do problema,
     ou apenas reproduz trigonometria orbital básica?".
   - Definição do baseline:
     Considerando que a órbita da Terra é aproximadamente circular com raio r_terra ≈ 1.0 AU:
       * Se o periélio do asteroide q > 1.0 AU: o asteroide NUNCA cruza a órbita da Terra.
         A menor distância radial geométrica possível no espaço entre as órbitas é (q - 1.0) AU.
       * Se q <= 1.0 AU (e o afélio Q >= 1.0 AU): a órbita do asteroide intersecta a
         esfera de 1 AU da Terra, logo a aproximação planar teórica mínima é 0.0 AU.
     Portanto:
       y_pred_baseline = max(q - 1.0, 0.0)
   - Uso:
     Calcular MAE, RMSE e R² (via `src.comum.metricas.calcular_metricas`) para servir
     como régua mínima de qualidade para qualquer modelo preditivo.

2. MÉTRICAS ESTRATIFICADAS POR REGIÃO DE MOID:
   - Motivação científica:
     Métricas globais (como MAE médio no dataset inteiro) são fortemente enviesadas pela
     enorme massa de asteroides distantes (MOID > 1 AU), que não oferecem risco algum.
     Um modelo com MAE global aparentemente baixo pode errar grosseiramente nos asteroides
     mais próximos da Terra (onde o erro é mais crítico).
   - Faixas sugeridas para estratificação:
       a) Faixa Crítica (PHA - Potentially Hazardous Asteroids): MOID <= 0.05 AU
          (~19.5 distâncias lunares). Limiar oficial da NASA/JPL para asteroides perigosos.
       b) Faixa de Atenção / Proximidade Moderada: 0.05 < MOID <= 0.15 AU.
       c) Faixa Segura / Distante: MOID > 0.15 AU.
   - O que implementar:
     Função que recebe y_true e y_pred, divide nas faixas acima e calcula N, MAE,
     RMSE e R² para cada faixa isoladamente, retornando um DataFrame comparativo.

3. IMPORTÂNCIA DE FEATURES (PERMUTATION IMPORTANCE):
   - Extrair a importância das features (originais vs novas features físicas)
     usando permutation_importance do scikit-learn ou a importância nativa do modelo.
   - Verificar se as novas variáveis (q, r_plus, r_minus, T_J) superam as variáveis
     brutas (a, e, i, om, w) no ganho preditivo.
"""


def calcular_baseline_geometrico(q):
    """TODO: Retornar np.maximum(q - 1.0, 0.0) como cota inferior analítica."""
    pass


def avaliar_baseline_geometrico(y_true, q):
    """TODO: Calcular MAE, RMSE e R² do baseline geométrico contra o y_true (MOID real em AU)."""
    pass


def metricas_por_regiao_moid(y_true, y_pred, faixas=None):
    """TODO: Estratificar as métricas por faixa de proximidade:
    - PHA: MOID <= 0.05 AU
    - Moderado: 0.05 < MOID <= 0.15 AU
    - Seguro: MOID > 0.15 AU
    Retornar DataFrame com contagem de amostras, MAE, RMSE e R² por faixa.
    """
    pass


def calcular_feature_importance(modelo_ou_predictor, X_test, y_test=None):
    """TODO: Calcular permutation importance ou ranking de ganho das features orbitais e físicas."""
    pass
