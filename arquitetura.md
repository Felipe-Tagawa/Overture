# Overture: Proposta de Nova Arquitetura

Criado para organizar o repositório e separar as duas frentes de trabalho: a frente congelada (legado SO) e a frente ativa (IC).

### 1 Árvore atual completa
```text
Overture/
├── .gitignore                                   [git]
├── LICENSE                                      [git]
├── README.md                                    [git]      mistura SO (concorrência, hardware) com ML/astronomia
├── requirements.txt                             [git]      mistura deps do legado (autogluon, xgboost, joblib) com o núcleo
├── astronomical_context.md                      [git]      documentação física solta na raiz
├── benchmark_overture.html                      [git, gerado por src/results/plot.py]  resultado do legado na raiz
├── apresentacao/
│   └── slides_ic_asteroides.html                [git]
│
├── AutogluonModels/                             [ignorado]  727 MB
│   ├── ag-20260902_030011/                      110 MB  execução avulsa (tabular_predictor.py, sem path fixo)
│   ├── moid_parallel/                           112 MB  execução antiga (antes do benchmark por time limit)
│   ├── moid_sequential/                         109 MB  execução antiga
│   ├── moid_parallel_25s/   moid_parallel_50s/   moid_parallel_100s/     benchmark oficial
│   └── moid_sequential_25s/ moid_sequential_50s/ moid_sequential_100s/  
│
└── src/
    ├── main.py                                  [git]
    ├── data/
    │   ├── __init__.py                          [git]
    │   ├── config.py                            [git]      features + target + caminhos no mesmo lugar; LABEL "moid_log" está em outro arquivo
    │   ├── extract.py                           [git]
    │   ├── load.py                              [git]
    │   ├── schema.toml                          [git]
    │   ├── transform.py                         [git]      responsabilidades: parse, validação, filtro de escopo, transformação do alvo
    │   ├── cache/asteroid_dataset.parquet       [ignorado] 70 MB
    │   └── report/
    │       ├── profile_report.py                [git]      QUEBRADO: `from src.data.transform import df` (df não existe)
    │       └── profile_report.html              [git, gerado] 3,5 MB
    ├── results/
    │   ├── plot.py                              [git]      código dentro da pasta de resultados; grava o HTML na raiz
    │   ├── autogluon_comparative_models.csv     [git]      resultado oficial do benchmark
    │   └── autogluon_comparative_models.parquet [git]      idêntico ao CSV
    ├── runners/
    │   ├── __init__.py                          [git]
    │   ├── benchmark.py                         [git]      split + treino + avaliação + prints + persistência no mesmo arquivo
    │   └── auto_gluon/
    │       └── tabular_predictor.py             [git]      QUEBRADO (dataset_client); duplica benchmark.py
    └── testes/                                             Nome sugere "testes automatizados", mas são experimentos
        ├── sequential_runner.py                 [git]  ┐
        ├── threaded_runner.py                   [git]  │   4 arquivos quase idênticos:
        ├── sequential_runner_xgb.py             [git]  │    {sequencial, threads} × {RFR, XGBoost}
        ├── thread_runner_xgb.py                 [git]  ┘
        └── models/
            ├── __init__.py                      [git]
            ├── HGBoostR.py                      [git]  ┐ QUEBRADOS (dataset_client)
            ├── RFR.py                           [git]  │ carregamento + split + métricas repetidos nos 3
            └── XGBoost.py                       [git]  ┘ RFR faz get_dummies em 'class', que não está em FEATURES
```

## 2. Nova arquitetura

### Legenda

| Marca | Significado |
|---|---|
| 📄 | Mantido como está (mesmo lugar) |
| ➡️ | Movido/renomeado, **sem** mudança de conteúdo (só imports) |
| ✂️ | **Extraído**: conteúdo vindo da separação de um arquivo maior |
| 🔗 | **Combinado**: junção de 2+ arquivos atuais em um |
| ✏️ | Mantido, com ajuste pequeno (imports, caminho, correção de bug) |
| 🆕 | Sugerido para os próximos passos (**não criado**; apenas a base) |
| 🗑️ | Remover do repositório/disco (conteúdo absorvido ou órfão) |
| 🔒 | Pertence à frente congelada (legado SO) |


```text
Overture/
├── README.md                          ✂️ visão geral do projeto (IC); o conteúdo de SO sai para docs/legado_so/
├── LICENSE                            📄
├── .gitignore                         ✏️ + artifacts/ ; remove regras de arquivos que já não existem na raiz
├── requirements.txt                   ✂️ só núcleo + IC (numpy, pandas, pandera, kagglehub, pyarrow, scikit-learn, matplotlib, pytest)
├── requirements-legado-so.txt         ✂️ -r requirements.txt + autogluon.tabular, xgboost, joblib, plotly
│
├── docs/                              🆕 pasta de documentação
│   ├── contexto_astronomico.md        ➡️ de astronomical_context.md
│   ├── legado_so/
│   │   ├── README.md                  ✂️ do README atual: Escopo SO, Conceitos de concorrência, Metodologia experimental,
│   │   │                                 Hardware dos integrantes, Modelos avaliados, fit_strategy
│   │   └── resultados.md              🆕 resumo congelado: tabela final do benchmark + melhor modelo + como reproduzir
│   └── ic/
│       └── roadmap.md                 🆕 etapas da IC (EDA → baseline físico → regressão → PHA → escala)
│
├── apresentacao/
│   └── slides_ic_asteroides.html      📄
│
├── src/
│   ├── core/                          ── NÚCLEO COMPARTILHADO (usado pelas duas frentes) ──
│   │   ├── __init__.py                ➡️ de src/data/__init__.py
│   │   ├── config/
│   │   │   ├── __init__.py            🆕
│   │   │   ├── paths.py               ✂️ de data/config.py (BASE_DIR, CACHE_PATH, SCHEMA_PATH, RESULTS_PATH → artifacts/)
│   │   │   └── dataset.py             🔗 FEATURES/TARGET/DISPLAY_COLUMNS de data/config.py + LABEL de benchmark.py
│   │   │                                 e de tabular_predictor.py (fonte única de nomes de colunas)
│   │   ├── data/
│   │   │   ├── __init__.py            🆕
│   │   │   ├── extract.py             ➡️ de src/data/extract.py
│   │   │   ├── load.py                ✏️ de src/data/load.py (import de paths)
│   │   │   ├── schema.toml            ➡️ de src/data/schema.toml
│   │   │   ├── validate.py            ✂️ de transform.py: TYPE_MAP, build_pandera_schema, parse Y/N→bool, to_numeric, dropna
│   │   │   └── pipeline.py            ✂️ de main.py: get_clean_data() (cache → extract → validate → save)
│   │   └── evaluation/
│   │       ├── __init__.py            🆕
│   │       └── metrics.py             🔗 bloco "expm1 + MAE/RMSE/R²" de benchmark.py, tabular_predictor.py,
│   │                                     HGBoostR.py, RFR.py e XGBoost.py → regression_report(y_true, y_pred_log)
│   │
│   ├── legado_so/                     🔒 ── FRENTE 1: ORIGEM EM SISTEMAS OPERACIONAIS (CONGELADA) ──
│   │   ├── __init__.py                🆕
│   │   ├── __main__.py                ✂️ de main.py: bloco `if __name__ == "__main__"` → `python -m src.legado_so`
│   │   ├── scope.py                   ✂️ de transform.py: filtro a<5, e<0.9 + log1p **congelados** como foram usados
│   │   ├── autogluon/
│   │   │   ├── __init__.py            ➡️ de src/runners/__init__.py
│   │   │   ├── benchmark.py           ✂️ de runners/benchmark.py: make_predictor, train_and_evaluate, run_benchmark
│   │   │   ├── report.py              ✂️ de runners/benchmark.py: print_detailed_models, print_benchmark_summary
│   │   │   └── plot.py                ✏️➡️ de src/results/plot.py (saída do HTML → artifacts/legado_so/)
│   │   ├── threading/
│   │   │   ├── __init__.py            ➡️ de src/testes/models/__init__.py
│   │   │   ├── models.py              🔗 fábricas make_model_rf / make_model_xgb / make_model_hgb
│   │   │   │                             de RFR.py + XGBoost.py + HGBoostR.py (sem carregar dados no import)
│   │   │   └── runner.py              🔗 sequential_runner.py + threaded_runner.py + sequential_runner_xgb.py
│   │   │                                 + thread_runner_xgb.py → `--modelo {rf,xgb,hgb} --estrategia {seq,thread}`
│   │   └── results/                   🔒 resultados oficiais versionados (somente leitura)
│   │       ├── autogluon_comparative_models.csv      ➡️ de src/results/
│   │       ├── autogluon_comparative_models.parquet  ➡️ de src/results/
│   │       └── benchmark_overture.html               ➡️ da raiz
│   │
│   └── ic/                            🚀 ── FRENTE 2: ESCALONAMENTO / INICIAÇÃO CIENTÍFICA (ATIVA) ──
│       ├── __init__.py                🆕
│       ├── __main__.py                🆕 ponto de entrada da IC → `python -m src.ic`
│       ├── scope.py                   ✂️ de transform.py: filtros de população **parametrizados** (ponto de partida = mesmos
│       │                                 filtros do legado; a EDA decide os novos)
│       ├── target.py                  ✂️ de transform.py: transformação do alvo (log1p hoje; espaço para log(x+ε))
│       ├── features.py                🆕 features físicas: q, Q, sin/cos(Ω, ω), distâncias nodais r±, Tisserand T_J
│       ├── baselines.py               🆕 baselines: max(q−1, 0) e DummyRegressor
│       ├── evaluation.py              ✂️ de tabular_predictor.py: feature_importance (permutation) +
│       │                                 🆕 métricas por região de MOID (usa core/evaluation/metrics.py)
│       └── eda/
│           └── profile_report.py      ✏️➡️ de src/data/report/profile_report.py (bug D2 corrigido; saída → artifacts/ic/)
│
├── tests/                             🆕 testes automatizados reais (pytest já está nas deps)
│   ├── test_validate.py               🆕 schema pandera aceita/rejeita linhas conhecidas
│   ├── test_metrics.py                🆕 regression_report com casos triviais
│   └── test_legado_so_reprodutivel.py 🆕 compara os números do CSV congelado com uma tolerância (smoke test)
│
└── artifacts/                         🆕 [ignorado] tudo o que é GERADO e pesado, fora de src/
    ├── cache/
    │   └── asteroid_dataset.parquet   ➡️ de src/data/cache/
    ├── legado_so/
    │   └── autogluon_models/          ➡️ de AutogluonModels/{moid_*_25s, _50s, _100s} (o benchmark oficial)
    └── ic/
        └── reports/
            └── profile_report.html    ➡️ de src/data/report/ (gerado; sai do git)
```


### 2.2 Itens que saem

```text
🗑️ src/main.py                                  → ✂️ dividido em core/data/pipeline.py + legado_so/__main__.py
🗑️ src/data/config.py                           → ✂️ dividido em core/config/paths.py + core/config/dataset.py
🗑️ src/data/transform.py                        → ✂️ dividido em core/data/validate.py + legado_so/scope.py + ic/scope.py + ic/target.py
🗑️ src/runners/benchmark.py                     → ✂️ dividido em legado_so/autogluon/benchmark.py + report.py
🗑️ src/runners/auto_gluon/tabular_predictor.py  → 🔗 absorvido: métricas → core/evaluation/metrics.py;
                                                     feature_importance → ic/evaluation.py; treino já coberto por benchmark.py
🗑️ src/testes/models/{RFR,XGBoost,HGBoostR}.py  → 🔗 legado_so/threading/models.py (+ métricas → core)
🗑️ src/testes/*_runner*.py (4 arquivos)         → 🔗 legado_so/threading/runner.py
🗑️ src/results/ (pasta)                         → código ➡️ legado_so/autogluon/plot.py; dados ➡️ legado_so/results/
🗑️ astronomical_context.md (raiz)               → ➡️ docs/contexto_astronomico.md
🗑️ benchmark_overture.html (raiz)               → ➡️ src/legado_so/results/
🗑️ data/  models/  runners/  (raiz, órfãs)      → apagar: só __pycache__ e um parquet duplicado de 64 MB
🗑️ AutogluonModels/{ag-20260902_030011, moid_parallel, moid_sequential}
                                                → execuções fora do benchmark oficial (~330 MB); arquivar fora do projeto ou apagar
```


### 2.3 Mapa origem -> destino (todos os arquivos versionados)

| Arquivo atual | Operação | Destino proposto |
|---|---|---|
| `.gitignore` | ✏️ | `.gitignore` |
| `LICENSE` | 📄 | `LICENSE` |
| `README.md` | ✂️ | `README.md` (IC) + `docs/legado_so/README.md` (SO) |
| `requirements.txt` | ✂️ | `requirements.txt` + `requirements-legado-so.txt` |
| `astronomical_context.md` | ➡️ | `docs/contexto_astronomico.md` |
| `benchmark_overture.html` | ➡️ | `src/legado_so/results/benchmark_overture.html` |
| `apresentacao/slides_ic_asteroides.html` | 📄 | `apresentacao/slides_ic_asteroides.html` |
| `src/main.py` | ✂️ | `src/core/data/pipeline.py` + `src/legado_so/__main__.py` |
| `src/data/__init__.py` | ➡️ | `src/core/__init__.py` |
| `src/data/config.py` | ✂️🔗 | `src/core/config/paths.py` + `src/core/config/dataset.py` |
| `src/data/extract.py` | ➡️ | `src/core/data/extract.py` |
| `src/data/load.py` | ✏️ | `src/core/data/load.py` |
| `src/data/schema.toml` | ➡️ | `src/core/data/schema.toml` |
| `src/data/transform.py` | ✂️ | `src/core/data/validate.py` + `src/legado_so/scope.py` + `src/ic/scope.py` + `src/ic/target.py` |
| `src/data/report/profile_report.py` | ✏️➡️ | `src/ic/eda/profile_report.py` |
| `src/data/report/profile_report.html` | ➡️ | `artifacts/ic/reports/profile_report.html` (gerado, sai do git) |
| `src/results/plot.py` | ✏️➡️ | `src/legado_so/autogluon/plot.py` |
| `src/results/autogluon_comparative_models.csv` | ➡️ | `src/legado_so/results/` |
| `src/results/autogluon_comparative_models.parquet` | ➡️ | `src/legado_so/results/` |
| `src/runners/__init__.py` | ➡️ | `src/legado_so/autogluon/__init__.py` |
| `src/runners/benchmark.py` | ✂️ | `src/legado_so/autogluon/benchmark.py` + `report.py` (+ métricas → `core/evaluation/metrics.py`) |
| `src/runners/auto_gluon/tabular_predictor.py` | 🔗 | `core/evaluation/metrics.py` + `ic/evaluation.py` |
| `src/testes/models/__init__.py` | ➡️ | `src/legado_so/threading/__init__.py` |
| `src/testes/models/HGBoostR.py` | 🔗 | `src/legado_so/threading/models.py` |
| `src/testes/models/RFR.py` | 🔗 | `src/legado_so/threading/models.py` |
| `src/testes/models/XGBoost.py` | 🔗 | `src/legado_so/threading/models.py` |
| `src/testes/sequential_runner.py` | 🔗 | `src/legado_so/threading/runner.py` |
| `src/testes/threaded_runner.py` | 🔗 | `src/legado_so/threading/runner.py` |
| `src/testes/sequential_runner_xgb.py` | 🔗 | `src/legado_so/threading/runner.py` |
| `src/testes/thread_runner_xgb.py` | 🔗 | `src/legado_so/threading/runner.py` |

**Resumo:** 30 arquivos versionados hoje. Na proposta:
- 6 são extraídos (✂️);
- 8 são combinados (🔗) em `metrics.py`, `models.py` e `runner.py`, com parte do `tabular_predictor.py` indo para `ic/evaluation.py`;
- 14 são movidos ou ajustados (➡️/✏️);
- 2 ficam como estão (📄).

Os novos (🆕) são só a base mínima pro nosso escalonamento.