"""Módulo de ingestão, validação e persistência do dataset de asteroides.

Responsável por extrair o dataset bruto, aplicar regras de tipo e esquema Pandera,
e gerenciar o cache local em formato Parquet.
"""

import tomllib
from pathlib import Path
import pandas as pd
import pandera.pandas as pa

from src.comum.config import (
    FEATURES,
    TARGET,
    FINAL_COLUMN,
    DISPLAY_COLUMNS,
    SCHEMA_PATH,
    CACHE_PATH,
)

TYPE_MAP = {
    "string": pa.String,
    "float": pa.Float,
    "integer": pa.Int64,
    "boolean": pa.BOOL,
}

NULLABLE_COLUMNS = {"pha", "neo"}


def extract_raw_dataset() -> pd.DataFrame:
    """Baixa e extrai o dataset bruto de asteroides via KaggleHub."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    raw_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sakhawat18/asteroid-dataset",
        "dataset.csv",
        pandas_kwargs={"low_memory": False},
    )
    return raw_df


def save_to_parquet(df: pd.DataFrame, path: Path = CACHE_PATH) -> None:
    """Salva o DataFrame em formato Parquet no caminho especificado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Dados salvos com sucesso em: {path}")


def load_from_parquet(path: Path = CACHE_PATH) -> pd.DataFrame:
    """Carrega dados salvos em arquivo Parquet."""
    print(f"Carregando dados do cache: {path}")
    return pd.read_parquet(path)


def cache_exists(path: Path = CACHE_PATH) -> bool:
    """Verifica se o arquivo de cache existe."""
    return path.exists()


def build_pandera_schema(schema_path: Path = SCHEMA_PATH) -> pa.DataFrameSchema:
    """Constrói o esquema de validação Pandera a partir do arquivo schema.toml."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Arquivo de Schema não encontrado: {schema_path}")

    with open(schema_path, "rb") as f:
        config = tomllib.load(f)

    column_rules = config.get("columns", {})
    validation_fields = {}

    for col in FEATURES + [TARGET] + FINAL_COLUMN + DISPLAY_COLUMNS:
        if col in column_rules:
            toml_type = column_rules[col]["type"]
            pandera_type = TYPE_MAP.get(toml_type, pa.Float)

            check = [pa.Check.gt(0)] if col in {"a", "moid"} else []
            nullable = col in NULLABLE_COLUMNS

            validation_fields[col] = pa.Column(
                dtype=pandera_type, checks=check, nullable=nullable, coerce=True
            )
        else:
            print(f"Aviso: Coluna '{col}' não configurada no schema.")

    return pa.DataFrameSchema(columns=validation_fields, strict="filter")


def validate_data(raw_df: pd.DataFrame, schema_path: Path = SCHEMA_PATH) -> pd.DataFrame:
    """Normaliza tipos booleanos/numéricos e valida o DataFrame contra o esquema Pandera."""
    df = raw_df.copy()

    for bool_col in ("pha", "neo"):
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map({"Y": True, "N": False})

    for col in FEATURES + [TARGET]:
        if col in df.columns and col not in NULLABLE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in FEATURES + [TARGET] if c in df.columns])

    schema = build_pandera_schema(schema_path)
    try:
        validated_df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        invalid_indexes = e.failure_cases["index"].dropna().unique()
        df = df.drop(index=invalid_indexes)
        validated_df = schema.validate(df, lazy=False)

    return validated_df


def get_clean_data(force_reprocess: bool = False, cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    """Obtém o dataset limpo e validado (lendo do cache ou processando do zero)."""
    if cache_exists(cache_path) and not force_reprocess:
        print("Cache encontrado, pulando etapas iniciais!")
        return load_from_parquet(cache_path)

    raw_df = extract_raw_dataset()
    clean_df = validate_data(raw_df)
    save_to_parquet(clean_df, cache_path)

    return clean_df
