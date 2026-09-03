# Overture - Multithreads em Machine Learning (Análises de Asteroides)

Projeto dedicado à primeira etapa da disciplina de Sistemas Operacionais (C12).

## Integrantes

Alunos: Pedro Henrique Ribeiro Dias e Felipe Tagawa Reis
Orientador: Jonas Lopes de Vilas Boas
Co-orientador: Felipe Augusto

## Status Atual
---

Dataset, tarefa de ML e abordagem de modelagem já definidos. O projeto está na fase de **experimentação de paralelismo**: comparação de tempo/desempenho entre treino sequencial e treino paralelo (threads e, no caso do AutoML, o mecanismo interno de paralelismo do AutoGluon).

## Escopo Geral
---

O projeto aplica conceitos de **paralelismo (threads/multithreading)** ao treinamento de modelos preditivos de Machine Learning, com dois objetivos centrais:

1. **Comparar desempenho de execução** (tempo de treino) entre treinar modelos de forma **sequencial** (um de cada vez) e de forma **paralela via threads** (vários ao mesmo tempo), usando o mesmo hardware (varia de acordo com as especificações pessoais do hardware utilizado para treinar os modelos).
2. **Usar esse paralelismo para treinar múltiplos modelos candidatos e compará-los**, funcionando como uma espécie de *benchmark*, não com foco em obter necessariamente o menor tempo de treino absoluto, mas em entender **como o paralelismo se comporta** ao treinar cargas de trabalho de Machine Learning (**Observação**: método utilizado anteriormente à aplicação do AutoGluon).

> **Nota:** o objetivo não é obter o melhor desempenho de hardware possível (como aconteceria isolando um único modelo em toda a máquina). O objetivo é observar o **efeito de rodar múltiplos treinos concorrentemente**, dividindo os recursos da máquina entre eles, que é exatamente o tipo de cenário estudado em Sistemas Operacionais ao lidar com concorrência e compartilhamento de CPU. Principalmente tendo em vista que o hardware pessoal disponível para o projeto é limitado (máquinas pessoais, com poucos núcleos e threads no processdor).

## Tarefa de Machine Learning
---

- **Tipo de tarefa:** regressão.
- **Alvo (target):** `MOID` (*Minimum Orbit Intersection Distance* — a menor distância entre a órbita de um asteroide e a órbita da Terra). É a métrica usada para avaliar o risco de colisão de um objeto próximo à Terra.
- **Transformação aplicada:** o alvo é treinado em escala logarítmica (`log1p`) para lidar com a forte assimetria da distribuição de distâncias orbitais, e revertido (`expm1`) antes de calcular as métricas finais, para que os resultados sejam interpretáveis na escala original (em AU ou LD — *Lunar Distance*).
- **Principais features utilizadas:** elementos orbitais keplerianos do asteroide — `a` (semi-eixo maior), `e` (excentricidade), `i` (inclinação), `om` (longitude do nodo ascendente), `w` (argumento do perihélio) — além de `H` (magnitude absoluta) e `pha` (flag de "Potentially Hazardous Asteroid").

Para mais detalhes sobre as features e a target, consulte o arquivo `astronomical_context.md` (documentação do vocabulário técnico da área de mecânica orbital/astrofísica).

## Dataset
---

- **Fonte:** Asteroid Dataset (Kaggle) — dados orbitais de objetos catalogados do Sistema Solar (principalmente NEOs — *Near-Earth Objects*).
- Um `schema.toml` documenta cada uma das colunas e uma breve explicação do seu significado astronômico.

## Modelos avaliados
---

Os modelos avaliados são definidos através do `auto_gluon`, que treina um portfólio de modelos candidatos e combina os melhores em um ensemble final. O portfólio inclui (principais modelos):
- **Random Forest** (`RandomForestRegressor`)
- **CatBoost** (`CatBoostRegressor`)
- **XGBoost** (`XGBRegressor`)
- **Weighted Ensemble L2** (combina os melhores modelos do portfólio em um ensemble final)
- **LightGBM** (`LGBMRegressor`)
- **LightGBMXT** (`LGBMXTRegressor`)
- **LightGBMLarge** (`LGBMRegressor` com `boosting_type="goss"`)
- **ExtraTreesMSE** (`ExtraTreesRegressor` com `criterion="mse"`)
- **NeuralNetFastAI** (rede neural via FastAI)


## Explicações técnicas dos termos utilizados
---

Esta seção existe porque o projeto cruza duas áreas, **Sistemas Operacionais** e **Machine Learning**, e alguns termos usados no código e nos relatórios podem não ser familiares a quem não é da área.

### Conceitos de concorrência (Sistemas Operacionais)

- **Thread:** uma linha de execução dentro de um processo. Várias threads de um mesmo processo compartilham a mesma memória, o que torna a comunicação entre elas mais barata que entre processos separados, mas também exige cuidado com acesso concorrente a dados compartilhados.
- **Processo:** uma instância independente de um programa em execução, com seu próprio espaço de memória. Processos diferentes não compartilham memória diretamente (precisam de mecanismos como pipes, sockets ou memória compartilhada explícita).
- **GIL (Global Interpreter Lock):** um mecanismo do interpretador padrão do Python (CPython) que garante que **apenas uma thread execute bytecode Python por vez**, mesmo em máquinas com múltiplos núcleos. Isso historicamente limitava o ganho de desempenho de multithreading em Python para código *puro* Python (cálculo numérico, laços, etc.).
- **Por que threads ainda ajudam em ML mesmo com o GIL:** bibliotecas como scikit-learn, XGBoost e o motor por trás do AutoGluon são escritas em C/C++/Cython. Essas partes **liberam o GIL** durante os trechos de computação pesada (ex.: construção de árvores, operações matriciais), permitindo que múltiplas threads Python realmente executem esse código em paralelo, mesmo sob o GIL.
- **Python *free-threaded* (`Py_GIL_DISABLED`, PEP 703):** uma variante experimental do CPython (a partir da 3.13) que remove o GIL por completo, permitindo paralelismo real de bytecode Python entre threads. O projeto verifica essa flag (`sysconfig.get_config_var("Py_GIL_DISABLED")`) para registrar em qual modo o experimento está rodando, já que os resultados de desempenho podem diferir entre a build padrão (com GIL) e a build livre de GIL. (**Observação**: não é utilizado no AutoGluon, que ainda depende do GIL para seu runtime interno.)
- **Multithreading vs. multiprocessamento:** multithreading compartilha memória entre as linhas de execução (mais leve, mas sujeito ao GIL para código Python puro); multiprocessamento roda processos totalmente independentes (contorna o GIL, mas com mais overhead de memória e comunicação). Alguns frameworks de ML (como o AutoGluon, via Ray) usam multiprocessamento internamente mesmo quando a opção é descrita como "paralela".

### Conceitos de Machine Learning

- **Hiperparâmetro:** um parâmetro do modelo definido *antes* do treino (não aprendido a partir dos dados), como o número de árvores de um ensemble ou a taxa de aprendizado. Ajustar hiperparâmetros é uma das etapas centrais para melhorar o desempenho de um modelo.
- **Overfitting:** quando o modelo aprende padrões específicos demais dos dados de treino (incluindo ruído), perdendo capacidade de generalizar para dados novos. Métricas calculadas em um conjunto de teste separado (nunca visto durante o treino) ajudam a detectar isso.
- **Ensemble:** técnica que combina previsões de múltiplos modelos para obter um resultado mais robusto que qualquer modelo individual (ex.: Random Forest combina várias árvores; um *Weighted Ensemble* combina modelos de tipos diferentes).
- **Early stopping:** interrupção do treino antes do número máximo de iterações, quando a performance em um conjunto de validação para de melhorar — evita treino desnecessário e overfitting.
- **Permutation importance:** técnica para medir a importância de uma feature embaralhando seus valores e observando o quanto a performance do modelo piora. É usada quando o modelo não fornece importância de feature nativamente (caso do `HistGradientBoostingRegressor` e do modelo final do AutoGluon).
- **AutoML (*Automated Machine Learning*):** automação de etapas do pipeline de ML normalmente feitas manualmente — pré-processamento, escolha de algoritmo, otimização de hiperparâmetros e ensembling — treinando e comparando muitos modelos candidatos automaticamente.
- **AutoGluon:** framework de AutoML de código aberto (da AWS) usado neste projeto via `benchmark.py`. Ele treina um portfólio diverso de modelos (árvores, boosting, redes neurais), aplica *bagging* (treino em múltiplos folds) e combina os melhores modelos em um ensemble final. Internamente, ele usa a biblioteca **Ray** para gerenciar a distribuição de CPUs/GPUs entre os modelos treinados.
- **`fit_strategy` (AutoGluon):** parâmetro nativo que decide se os modelos do portfólio do AutoGluon são treinados um de cada vez (`"sequential"`, padrão) ou simultaneamente, dividindo os recursos de CPU entre eles via Ray (`"parallel"`, ainda experimental). É o principal ponto de comparação entre execução sequencial e paralela no contexto do AutoML deste projeto.

## Metodologia experimental
---

Para cada modelo (exceto onde indicado), o desempenho é comparado em dois cenários:

1. **Sequencial:** N modelos treinados um após o outro, cada um usando todos os núcleos disponíveis internamente (`n_jobs`/`num_cpus` máximo).
2. **Paralelo (threads):** os mesmos N modelos treinados simultaneamente, dividindo os núcleos disponíveis entre eles (ex.: 4 modelos rodando ao mesmo tempo, cada um limitado a uma fração dos núcleos totais).

Para o AutoGluon, como treinar múltiplas instâncias completas simultaneamente via threads Python arrisca conflito no runtime interno do Ray (que não suporta múltiplas inicializações no mesmo processo), a comparação sequencial vs. paralelo é feita através do parâmetro nativo `fit_strategy`, aplicado a uma única instância do `TabularPredictor` por execução.

Em todos os casos, mede-se o **tempo total de treino** (`time.perf_counter()`) e, quando aplicável, as métricas de qualidade do modelo (**MAE**, **RMSE**, **R²**) — para verificar se o paralelismo afeta não só o tempo, mas também a qualidade do resultado final.

- **MAE** (*Mean Absolute Error*): média das diferenças absolutas entre as previsões e os valores reais. Mede o erro médio em unidades da variável alvo.
- **RMSE** (*Root Mean Squared Error*): raiz quadrada da média dos erros quadrados. Penaliza mais fortemente grandes erros, sendo sensível a outliers.
- **R²** (*Coefficient of Determination*): mede a proporção da variância dos dados que é explicada pelo modelo. Varia de 0 a 1, sendo 1 um modelo perfeito.

## Especificações de Hardware dos Integrantes
---

**Pedro Henrique Ribeiro Dias**
- **CPU:** Intel Core i5-11400H (6 núcleos, 12 threads, 2.7 GHz base, 4.5 GHz turbo)
- **RAM:** 16 GB DDR4
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB GDDR6)
- **Sistema Operacional:** Windows 11 (25H2)

**Felipe Tagawa Reis**
- **CPU:** AMD Ryzen 7 5800H (8 núcleos, 16 threads, 3.2 GHz base, 4.4 GHz turbo)
- **RAM:** 24 GB DDR4
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB GDDR6)
- **Sistema Operacional:** Linux Fedora 44

## Tecnologias utilizadas
---

- **Python** (incluindo testes com build *free-threaded*, PEP 703)
- **scikit-learn** — RandomForestRegressor, HistGradientBoostingRegressor, métricas, `permutation_importance`, `train_test_split`
- **XGBoost** — XGBRegressor
- **AutoGluon** — `TabularPredictor` (AutoML)
- **joblib** — orquestração de paralelismo (`Parallel`, `delayed`, `parallel_config`)
- **Ray** — motor de paralelismo interno usado pelo AutoGluon
- **pandas** — manipulação de DataFrames
- **numpy** — operações numéricas
- **matplotlib** — visualização de gráficos
- **pyarrow** — leitura/escrita de arquivos Parquet
- **panderas** — serialização de objetos Python (modelos treinados, resultados)
- **ydata-profiling** — análise exploratória de dados (EDA)
- **kagglehub** — download do dataset do Kaggle via API