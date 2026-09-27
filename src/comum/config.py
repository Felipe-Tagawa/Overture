from pathlib import Path

# Features orbitais utilizadas no problema
FEATURES = [
    "a",
    "e",
    "i",
    "om",
    "w",
]

# Alvo original (escala linear em AU)
TARGET = "moid"

# Alvo transformado utilizado para treinamento e convergência dos modelos
LABEL = "moid_log"

# Colunas adicionais e de exibição do dataset
FINAL_COLUMN = ["H"]
DISPLAY_COLUMNS = ["full_name", "pha", "moid_ld"]

# Caminhos do projeto
COMUM_DIR = Path(__file__).resolve().parent
SRC_DIR = COMUM_DIR.parent
ROOT_DIR = SRC_DIR.parent
BASE_DIR = ROOT_DIR

CACHE_PATH = COMUM_DIR / "cache" / "asteroid_dataset.parquet"
SCHEMA_PATH = COMUM_DIR / "schema.toml"
RESULTS_PATH = SRC_DIR / "legado_so" / "autogluon" / "resultados"
