from pathlib import Path

BASE_DIR = Path().resolve()

ROOT_DIR = BASE_DIR.parent

DATA_PATH = ROOT_DIR / "datasets" / "documents100.csv"