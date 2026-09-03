import tomllib
from pathlib import Path
import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.data.config import FEATURES, TARGET, FINAL_COLUMN, DISPLAY_COLUMNS, SCHEMA_PATH

TYPE_MAP = {
    "string": pa.String,
    "float": pa.Float,
    "integer": pa.Int64,
    "boolean": pa.BOOL,
}

NULLABLE_COLUMNS = {"pha", "neo"}

def build_pandera_schema() -> pa.DataFrameSchema:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Arquivo de Schema não encontrado: {SCHEMA_PATH}")

    with open(SCHEMA_PATH, "rb") as f:
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
            print("Aviso: Coluna '{col}' não configurada no schema.")

    return pa.DataFrameSchema(columns=validation_fields, strict="filter")

def transform_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for bool_col in ("pha", "neo"):
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map({"Y": True, "N": False})

    for col in FEATURES + [TARGET]:
        if col in df.columns and col not in NULLABLE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in FEATURES + [TARGET] if c in df.columns])

    schema = build_pandera_schema()
    try:
        validated_df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        invalid_indexes = e.failure_cases["index"].dropna().unique()
        df = df.drop(index=invalid_indexes)
        validated_df = schema.validate(df, lazy=False)

    df_clean = validated_df[
        (validated_df["a"] < 5) & (validated_df["e"] < 0.9)
        ].copy()

    if TARGET in df_clean.columns:
        df_clean["moid_log"] = np.log1p(df_clean[TARGET])

    return df_clean