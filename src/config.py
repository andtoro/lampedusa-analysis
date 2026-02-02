from pathlib import Path

# Root directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input data
MODEL_DATA_DIR = DATA_DIR / "new_data/out"
OBS_DATA_DIR = DATA_DIR / "new_data/obs"

# Output data
FIGURE_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "csv"

# Create output dirs if they don't exist
for d in [OUTPUT_DIR, FIGURE_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)