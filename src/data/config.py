from pathlib import Path
FEATURES = [
    "a",
    "e",
    "i",
    "om",
    "w",
]

TARGET = "moid"

FINAL_COLUMN = ["H"]

DISPLAY_COLUMNS = ["full_name", "pha", "moid_ld"]


BASE_DIR = Path(__file__).parent.parent
RESULTS_PATH = BASE_DIR / "results"
SCHEMA_PATH = BASE_DIR / "data" / "schema.toml"
CACHE_PATH = BASE_DIR / "data" / "cache" / "asteroid_dataset.parquet"