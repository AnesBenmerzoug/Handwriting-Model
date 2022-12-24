from pathlib import Path

__all__ = [
    "DATA_DIR",
    "OUTPUT_DIR",
    "TRANSCRIPTIONS_DIR",
    "LINE_STROKES_DIR",
    "PREPROCESSED_DATA_FILE",
]

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
TRANSCRIPTIONS_DIR = DATA_DIR / "ascii"
LINE_STROKES_DIR = DATA_DIR / "lineStrokes"
PREPROCESSED_DATA_FILE = DATA_DIR / "preprocessed_data.pickle"
