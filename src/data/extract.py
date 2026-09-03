import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def extract_raw_dataset() -> pd.DataFrame:
    raw_df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sakhawat18/asteroid-dataset",
        "dataset.csv",
        pandas_kwargs={"low_memory": False},
    )

    return raw_df