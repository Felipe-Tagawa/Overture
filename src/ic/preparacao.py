"""Módulo de Preparação de Dados e Engenharia de Features Físicas (Frente IC).

1. FILTRO DE ESCOPO PARAMETRIZÁVEL:
   - Diferente do legado (que tinha filtros a < 5 e e < 0.9 fixos no código),
     a IC precisa permitir testar diferentes populações orbitais (ex: Main Belt Asteroids,
     NEAs, ou cortes mais relaxados como a < 100 para cometas periódicos).
   - Deve receber parâmetros como min_a, max_a, max_e e retornar o subconjunto filtrado.

2. NOVAS FEATURES FÍSICAS PROPOSTAS PARA A IC:
   ---------------------------------------------------------------------------
   A partir dos 5 elementos orbitais keplerianos brutos (a, e, i, om, w):

   # Provavelmente Periélio e Afélio já existem no dataset

   a) Periélio ('q'):
      - Fórmula: q = a * (1 - e)
      - Unidade: Unidades Astronômicas (AU).
      - Significado físico: Distância mínima do asteroide ao Sol ao longo de sua órbita.
        Se q > 1 AU, o asteroide nunca entra na órbita da Terra. É a variável estrutural
        mais informativa para o MOID.

   b) Afélio ('Q'):
      - Fórmula: Q = a * (1 + e)
      - Unidade: Unidades Astronômicas (AU).
      - Significado físico: Ponto mais afastado do Sol na órbita. Útil para determinar
        se o asteroide cruza a órbita de planetas exteriores como Marte e Júpiter.

   c) Decomposição Trigonométrica dos Ângulos ('sin_om', 'cos_om', 'sin_w', 'cos_w'):
      - Fórmula:
          sin_om = np.sin(np.radians(om)), cos_om = np.cos(np.radians(om))
          sin_w  = np.sin(np.radians(w)),  cos_w  = np.cos(np.radians(w))
      - Significado físico: Os ângulos 'om' (longitude do nó ascendente) e 'w' (argumento
        do periélio) são coordenadas circulares periódicas [0°, 360°]. A descontinuidade
        entre 359° e 0° degrada modelos lineares e baseados em distância. A decomposição
        em seno e cosseno mapeia os ângulos em um espaço contínuo no círculo trigonométrico.

   d) Distâncias Nodais ('r_plus', 'r_minus'):
      - Fórmula:
          p = a * (1 - e^2)   (semi-latus rectum)
          r_plus  = p / (1 + e * np.cos(np.radians(w)))
          r_minus = p / (1 - e * np.cos(np.radians(w)))
      - Unidade: AU.
      - Significado físico: As distâncias heliocêntricas exatamente nos dois nós orbitais
        onde o plano da órbita do asteroide intersecta o plano da eclíptica da Terra
        (nó ascendente e nó descendente). Se um dos nós estiver muito próximo de 1 AU
        (distância da Terra ao Sol), o asteroide tem alto potencial de cruzamento com a Terra.

   e) Parâmetro de Tisserand em Relação a Júpiter ('T_J'):
      - Fórmula:
          T_J = (a_J / a) + 2 * np.cos(np.radians(i)) * np.sqrt((a / a_J) * (1 - e^2))
          onde a_J = 5.2044 AU (semi-eixo maior de Júpiter).
      - Significado físico: Invariante quasi-conservado do problema restrito de 3 corpos.
        Classifica dinamicamente o asteroide:
          * T_J > 3: Asteroides clássicos do Cinturão Principal (pouca influência joviana).
          * 2 < T_J < 3: Asteroides da família de cometas de Júpiter.
          * T_J < 2: Cometas quase-parabólicos / órbitas de longo período.
        Ajuda o modelo a entender o regime gravitacional do objeto.
"""

# Constante astronômica de referência para cálculos futuros
A_JUPITER = 5.2044  # Semi-eixo maior de Júpiter em AU


def filtrar_escopo(df, max_a=5.0, max_e=0.9, min_a=0.0):
    """TODO: Filtrar o DataFrame de acordo com os limites orbitais desejados para o experimento."""
    pass


def transformar_alvo(df, target_col="moid", label_col="moid_log", metodo="log1p", epsilon=1e-4):
    """TODO: Aplicar transformação do alvo (log1p ou log_eps com epsilon configurável)."""
    pass


def adicionar_features_fisicas(df):
    """TODO: Implementar o cálculo das novas features físicas orbitais:
    - q = a * (1 - e)
    - Q = a * (1 + e)
    - sin_om, cos_om = sin(om), cos(om)
    - sin_w, cos_w = sin(w), cos(w)
    - r_plus, r_minus = distâncias nodais (eclíptica)
    - T_J = parâmetro de Tisserand em relação a Júpiter
    """
    pass


def preparar_dados_ic(df, max_a=5.0, max_e=0.9, metodo_alvo="log1p"):
    """TODO: Orquestrar o pipeline completo:
    1. Filtrar escopo
    2. Adicionar features físicas
    3. Transformar o alvo
    """
    pass
