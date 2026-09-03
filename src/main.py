from src.data.extract import extract_raw_dataset
from src.data.transform import transform_data
from src.data.load import cache_exists, load_from_parquet, save_to_parquet
from src.runners.benchmark import run_benchmark

def get_clean_data(force_reprocess: bool = False):
    if cache_exists() and not force_reprocess:
        print("Cache encontrado, pulando etapas inicias!")
        return load_from_parquet()

    raw_df = extract_raw_dataset()
    clean_df = transform_data(raw_df)
    save_to_parquet(clean_df)

    return clean_df

if __name__ == "__main__":
    df_final = get_clean_data(force_reprocess=True) # Trocar para True quando for mudar o arquivo parquet
    print(f"Total de Registros: {len(df_final)}")
    print(f"Colunas Disponíveis: {list(df_final.columns)}")

    run_benchmark(df=df_final)