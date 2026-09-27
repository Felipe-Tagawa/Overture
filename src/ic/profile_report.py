"""Geração de Relatório Exploratório Automatizado (EDA via ydata-profiling).
"""

from pathlib import Path
import numpy as np
from ydata_profiling import ProfileReport

from src.comum.config import FEATURES, TARGET
from src.comum.dados import get_clean_data

OUTPUT_PATH = Path(__file__).parent / "profile_report.html"

df = get_clean_data(force_reprocess=False)
if "moid_log" not in df.columns and TARGET in df.columns:
    df["moid_log"] = np.log1p(df[TARGET])

columns_to_profile = FEATURES + [TARGET, "moid_log"]

if __name__ == "__main__":
    profile = ProfileReport(
        df[columns_to_profile],
        title="Asteroid Dataset — Profiling Report",
        explorative=True,
    )

    profile.to_file(OUTPUT_PATH)

    print(f"Relatório gerado em: {OUTPUT_PATH}")

# Vital: xdg-open /home/kyo/Repos/Overture/src/ic/profile_report.html --> Relatório ótimo para EDA
