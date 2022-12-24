import logging
import tarfile
from pathlib import Path

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    DATA_DIR = Path(__file__).parent.parent / "data"
    ASCII_DATA_FILE = DATA_DIR / "ascii-all.tar.gz"
    LINE_STROKES_DATA_FILE = DATA_DIR / "lineStrokes-all.tar.gz"

    ASCII_DATA_DIR = DATA_DIR / "ascii"
    LINE_STROKES_DATA_DIR = DATA_DIR / "lineStrokes"

    if not (ASCII_DATA_FILE.exists() and LINE_STROKES_DATA_FILE.exists()):
        raise FileNotFoundError(
            f"Could not find data files. "
            "Please make sure to follow the instructions in the README "
            "to download and set up the dataset."
        )

    if ASCII_DATA_DIR.exists():
        logger.info(f"{ASCII_DATA_DIR} directory already exists. Skipping")
    else:
        logger.info(f"Extracting {ASCII_DATA_FILE} into {ASCII_DATA_DIR}.")
        with tarfile.open(ASCII_DATA_DIR, "r") as tar:
            tar.extractall(path=DATA_DIR)

    if LINE_STROKES_DATA_DIR.exists():
        logger.info(f"{LINE_STROKES_DATA_DIR} directory already exists. Skipping")
    else:
        logger.info(
            f"Extracting {LINE_STROKES_DATA_FILE} into {LINE_STROKES_DATA_DIR}."
        )
        with tarfile.open(LINE_STROKES_DATA_DIR, "r") as tar:
            tar.extractall(path=DATA_DIR)
