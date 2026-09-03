# data/profile_report.py

from pathlib import Path
from ydata_profiling import ProfileReport

from src.data.config import FEATURES, TARGET
from src.data.transform import df

OUTPUT_PATH = Path(__file__).parent / "profile_report.html"

columns_to_profile = FEATURES + [TARGET, "moid_log"]

profile = ProfileReport(
    df[columns_to_profile],
    title="Asteroid Dataset — Profiling Report",
    explorative=True,
)

profile.to_file(OUTPUT_PATH)

print(f"Relatório gerado em: {OUTPUT_PATH}")

# Vital: xdg-open /home/kyo/Repos/Overture/data/profile_report.html --> Relatório ótimo para EDA