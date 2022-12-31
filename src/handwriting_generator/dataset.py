import logging
import pickle
import string

import numpy as np
import pytorch_lightning as pl
import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset, random_split

from handwriting_generator.constants import (
    LINE_STROKES_DIR,
    PREPROCESSED_DATA_FILE,
    TRANSCRIPTIONS_DIR,
)
from handwriting_generator.utils import (
    collate_fn,
    convert_stroke_set_to_array,
    filter_line_strokes_and_transcriptions,
    load_line_strokes,
    load_transcriptions,
)

__all__ = ["IAMDataset", "IAMDataModule"]

logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """IAM On-Line Handwriting Dataset Class"""

    def __init__(self, alphabet: str):
        self.alphabet = alphabet

        self.strokes_array_list: list[np.ndarray] = []
        self.transcriptions_list: list[str] = []
        self.transcriptions_onehot_list: list[np.ndarray] = []

        with PREPROCESSED_DATA_FILE.open("rb") as f:
            (
                self.strokes_array_list,
                self.transcriptions_list,
                self.transcriptions_onehot_list,
            ) = pickle.load(f)

        self.length = len(self.transcriptions_list)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        strokes = torch.tensor(self.strokes_array_list[idx])
        onehot = torch.tensor(self.transcriptions_onehot_list[idx])
        transcription = self.transcriptions_list[idx]
        return strokes, onehot, transcription


class IAMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_size: float = 0.8,
        test_size: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = 1 - (self.train_size + self.test_size)
        self.num_workers = num_workers
        if self.train_size < 0 or self.train_size >= 1.0:
            raise ValueError("train_size should be in the range (0, 1)")

        # unknown character + space + some punctuation marks + lowercase letters + uppercase letters
        self.unknown_character = "^"
        self.alphabet = (
            self.unknown_character
            + ' .,"'
            + string.ascii_lowercase
            + string.ascii_uppercase
        )

        if not (TRANSCRIPTIONS_DIR.exists() and LINE_STROKES_DIR.exists()):
            raise FileNotFoundError(
                f"Could not find data files. "
                "Please make sure to follow the instructions in the README "
                "to download and set up the dataset."
            )

    def prepare_data(self) -> None:
        if PREPROCESSED_DATA_FILE.exists():
            logger.info(
                f"Preprocessed data file {PREPROCESSED_DATA_FILE} exists already"
            )
            return
        transcriptions = load_transcriptions(TRANSCRIPTIONS_DIR)
        line_strokes = load_line_strokes(LINE_STROKES_DIR)
        line_strokes, transcriptions = filter_line_strokes_and_transcriptions(
            line_strokes, transcriptions
        )

        strokes_array_list = []
        transcriptions_list = []
        transcriptions_onehot_list = []

        for key in track(line_strokes.keys()):
            stroke_set = line_strokes[key]
            transcription = transcriptions[key]
            if len(transcription) <= 5:
                logger.info(f"Transcription is too short: '{transcription}'")
                continue
            elif len(transcription) >= 50:
                logger.info(f"Transcription is too long: '{transcription}'")
                continue

            strokes_array = convert_stroke_set_to_array(stroke_set)
            strokes_array_list.append(strokes_array)

            transcription = "".join(
                c if c in self.alphabet else self.alphabet[0] for c in transcription
            )
            onehot = np.zeros(
                shape=(len(transcription), len(self.alphabet) + 1), dtype=np.uint8
            )
            indices = [self.alphabet.find(c) for c in transcription]
            onehot[np.arange(len(transcription)), indices] = 1

            transcriptions_list.append(transcription)
            transcriptions_onehot_list.append(onehot)

        with open(PREPROCESSED_DATA_FILE, "wb+") as f:
            pickle.dump(
                [strokes_array_list, transcriptions_list, transcriptions_onehot_list], f
            )

    def setup(self, stage: str):
        dataset_full = IAMDataset(self.alphabet)
        subsets = random_split(
            dataset_full, [self.train_size, self.val_size, self.test_size]
        )
        self.train_dataset = subsets[0]
        self.val_dataset = subsets[1]
        self.test_dataset = subsets[2]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
