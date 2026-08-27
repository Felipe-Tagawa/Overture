import kagglehub
from kagglehub import KaggleDatasetAdapter
import tomllib
import pandera.pandas as pa
from data.config import FEATURES, TARGET
from pathlib import Path
import numpy as np

# TODO: Modularizar isso tudo aqui pelo amor de DEUS!!!!
# TODO: Fazer testes com as funcões criadas

SCHEMA_PATH = Path(__file__).parent / "schema.toml"

with open(SCHEMA_PATH, 'rb') as f: # tomllib exige 'rb' no lugar de 'r' - read apenas
    config = tomllib.load(f)

column_rules = config["columns"]

type_map = {
    "string": pa.String,
    "float": pa.Float,
    "integer": pa.Int64, # Evitar problemas com nulos
}

validation_field= {}

for column_name in FEATURES + [TARGET]:
    if column_name in column_rules:
        toml_type = column_rules[column_name]["type"]
        pandera_type = type_map.get(toml_type, pa.Float) # Valores reais 

        check = [pa.Check.gt(0)] if column_name in {"a", "moid"} else []

        validation_field[column_name] = pa.Column(dtype=pandera_type, checks=check, nullable=False, coerce=True)

    else:
        print(f"Aviso: Coluna '{column_name}' não foi encontrada no schema.toml")

# Validador do Pandera
schema_pandera = pa.DataFrameSchema(
    columns = validation_field,
    strict="filter",  # Remover outras colunas desnecessárias  
)

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sakhawat18/asteroid-dataset",
    "dataset.csv",
    pandas_kwargs={"low_memory": False}  # Análise completa dos tipos das colunas
)

try:
    df = schema_pandera.validate(df, lazy=True)
except pa.errors.SchemaErrors as e:
    invalid_indexes = e.failure_cases["index"].dropna().unique()
    df = df.drop(index=invalid_indexes)
    df = df[list(validation_field.keys())]  # Filtro de validaćão
    df = schema_pandera.validate(df, lazy=False)

# Remove outliers extremos (cometas quase-parabólicos)
df = df[(df["a"] < 100) & (df["e"] < 0.9)]

# Não há nulos nem duplicados (já testado)

df["moid_log"] = np.log1p(df[TARGET])

print(f"Colunas mantidas: {list(df.columns)}")
print(f"Total de registros prontos para treino: {len(df)}")


