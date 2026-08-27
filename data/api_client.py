import kagglehub
from kagglehub import KaggleDatasetAdapter
import tomllib
from IPython.display import display
import pandera as pa
from pandera.typing import Series

with open('schema.toml', 'rb') as f: # tomllib exige 'rb' no lugar de 'r' - read apenas
    config = tomllib.load(f)

class AsteroidsMetrics(pa.DataFrameModel):  
    id: Series[str]
    spkid: Series[str]
    full_name: Series[str]
    pdes: Series[str]
    name: Series[str]
    prefix: Series[str]
    neo: Series[bool]
    pha: Series[bool]
    H: Series[float]
    diameter: Series[float] = pa.Field(gt=0)
    albedo: Series[float] = pa.Field(gt=0)
    diameter_sigma: Series[float] = pa.Field(ge=0)
    





columns_rules = config["columns"]

obg_types = {
    'pdes': str,
    'name': str,
    'prefix': str
}

type_map = {
    "string": pa.String,
    "float": pa.Float,
    "integer": pa.Int,
    "boolean": pa.String # "Y" ou "N"
}

validation_field= {}

for column_name, properties in columns_rules.items():
    toml_type = properties["type"]
    pandera_type = type_map.get(toml_type, pa.String) # Padrão string
    
    validation_field[column_name] = pa.Column(pandera_type, nullable=True)

# Validador do Pandera
schema_pandera = pa.DataFrameSchema(validation_field)

initial_types = {'pdes': str, 'name': str, 'prefix': str}
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "sakhawat18/asteroid-dataset",
    "dataset.csv",
    dtype=initial_types
)


