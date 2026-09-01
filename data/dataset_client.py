import tomllib
import pandera.pandas as pa
from data.config import FEATURES, TARGET, FINAL_COLUMN, DISPLAY_COLUMNS
from pathlib import Path
import numpy as np
import pandas as pd

# TODO: Modularizar isso tudo aqui pelo amor de DEUS!!!!
# TODO: Fazer testes com as funcões criadas

SCHEMA_PATH = Path(__file__).parent / "schema.toml"
PARQUET_PATH = CACHE_PATH = Path(__file__).parent / "cache" / "asteroid_dataset.parquet"

with open(SCHEMA_PATH, 'rb') as f: # tomllib exige 'rb' no lugar de 'r' - read apenas
    config = tomllib.load(f)

column_rules = config["columns"]

type_map = {
    "string": pa.String,
    "float": pa.Float,
    "integer": pa.Int64, # Evitar problemas com nulos
    "boolean": pa.BOOL
}

NULLABLE_COLUMNS = {"pha", "neo"}

validation_field= {}

for column_name in FEATURES + [TARGET] + FINAL_COLUMN + DISPLAY_COLUMNS:
    if column_name in column_rules:
        toml_type = column_rules[column_name]["type"]
        pandera_type = type_map.get(toml_type, pa.Float) # Valores reais 

        check = [pa.Check.gt(0)] if column_name in {"a", "moid"} else []

        nullable = column_name in NULLABLE_COLUMNS

        validation_field[column_name] = pa.Column(dtype=pandera_type, checks=check, nullable=nullable, coerce=True)

    else:
        print(f"Aviso: Coluna '{column_name}' não foi encontrada no schema.toml")

# Validador do Pandera
schema_pandera = pa.DataFrameSchema(
    columns = validation_field,
    strict="filter",  # Remover outras colunas desnecessárias  
)

def download_n_validate():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    raw_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sakhawat18/asteroid-dataset",
    "dataset.csv",
    pandas_kwargs={"low_memory": False}  # Análise completa dos tipos das colunas
    )

    for bool_col in ("pha", "neo"):
        if bool_col in raw_df.columns:
            raw_df[bool_col] = raw_df[bool_col].map({"Y": True, "N": False})

    try:
        validated = schema_pandera.validate(raw_df, lazy=True)
    except pa.errors.SchemaErrors as e:
        invalid_indexes = e.failure_cases["index"].dropna().unique()
        raw_df = raw_df.drop(index=invalid_indexes)
        raw_df = raw_df[list(validation_field.keys())]  # Filtro de validaćão
        validated = schema_pandera.validate(raw_df, lazy=False)

    return validated

if CACHE_PATH.exists():
    df= pd.read_parquet(CACHE_PATH)
else:
    df = download_n_validate()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f"Dataset baixado e salvo em cache: {CACHE_PATH}")

# Remove outliers extremos (cometas quase-parabólicos)
df = df[(df["a"] < 5) & (df["e"] < 0.9)]

# Não há nulos nem duplicados (já testado)

df["moid_log"] = np.log1p(df[TARGET])

print(f"Colunas mantidas: {list(df.columns)}")
print(f"Total de registros prontos para treino: {len(df)}")


