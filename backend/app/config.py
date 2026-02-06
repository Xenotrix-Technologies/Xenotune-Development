from pathlib import Path

# Absolute path to backend/app/
BASE_DIR = Path(__file__).resolve().parent

# Runtime-writable config file
CONFIG_PATH = BASE_DIR.parent / "data" / "config.json"
