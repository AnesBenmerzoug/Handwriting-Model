import logging
import os
import pickle
import string

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from handwriting_generator.constants import DATA_DIR
from handwriting_generator.utils import (
    convert_stroke_set_to_array,
    filter_line_strokes_and_transcriptions,
    load_line_strokes,
    load_transcriptions,
)

__all__ = ["IAMDataset"]

logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """IAM On-Line Handwriting Dataset Class"""

    def __init__(self):
        self.transcriptions_dir = DATA_DIR / "ascii"
        self.line_strokes_dir = DATA_DIR / "lineStrokes"

        self.preprocessed_data_filename = DATA_DIR / "preprocessed_data.pickle"

        if not (self.transcriptions_dir.exists() and self.line_strokes_dir.exists()):
            raise FileNotFoundError(
                f"Could not find data files. "
                "Please make sure to follow the instructions in the README "
                "to download and set up the dataset."
            )

        # unknown character + space + some punctuation marks + lowercase letters + digits
        self.unknown_character = "^"
        self.alphabet = "".join(
            [self.unknown_character]
            + [" "]
            + [".,\"'?!:()"]
            + [c for c in string.ascii_lowercase + string.digits]
        )
        self.length = 0

        self.transcriptions: list[str] = []
        self.strokes_array_list: list[np.ndarray] = []
        self.transcriptions_onehot = []

        if not (os.path.exists(self.preprocessed_data_filename)):
            logger.info(
                f"Preprocessing data and caching it in file {self.preprocessed_data_filename}"
            )
            transcriptions_list, strokes_array_list = self.preprocess_data()
        else:
            logger.info(
                f"Preprocessed data file {self.preprocessed_data_filename} exists already"
            )
            transcriptions_list = None
            strokes_array_list = None

        self.load_data(transcriptions_list, strokes_array_list)

    def preprocess_data(self) -> tuple[list[str], list[np.ndarray]]:
        transcriptions = load_transcriptions(self.transcriptions_dir)
        line_strokes = load_line_strokes(self.line_strokes_dir)
        line_strokes, transcriptions = filter_line_strokes_and_transcriptions(
            line_strokes, transcriptions
        )

        transcriptions_list = []
        strokes_array_list = []

        with logging_redirect_tqdm():
            for key in tqdm(line_strokes.keys()):
                stroke_set = line_strokes[key]
                transcription = transcriptions[key]
                if len(transcription) <= 10:
                    logger.info(f"Transcription is too short: '{transcription}'")
                elif len(transcription) >= 50:
                    logger.info(f"Transcription is too long: '{transcription}'")
                else:
                    transcriptions_list.append(transcription)
                    strokes_array_list.append(convert_stroke_set_to_array(stroke_set))

        with open(self.preprocessed_data_filename, "wb+") as f:
            pickle.dump([transcriptions_list, strokes_array_list], f)

        return transcriptions_list, strokes_array_list

    def load_data(
        self,
        transcriptions_list: list[str] | None = None,
        strokes_array_list: list[np.ndarray] | None = None,
    ) -> None:
        if transcriptions_list is None or strokes_array_list is None:
            with self.preprocessed_data_filename.open("rb") as f:
                transcriptions_list, strokes_array_list = pickle.load(f)

        # Scale the X and Y components down by dividing by their standard deviation
        self.strokes_array_list = [
            np.concatenate([x[:, :2] / np.std(x[:, :2], axis=0), x[:, [2]]], axis=1)
            for x in strokes_array_list
        ]

        for transcription in tqdm(transcriptions_list):
            transcription = "".join(
                c if c in self.alphabet else self.unknown_character
                for c in transcription.lower()
            )
            onehot = np.zeros(
                shape=(len(transcription), len(self.alphabet) + 1), dtype=np.uint8
            )
            indices = [self.alphabet.find(c) for c in transcription]
            onehot[np.arange(len(transcription)), indices] = 1
            self.transcriptions.append(transcription)
            self.transcriptions_onehot.append(onehot)

        self.length = len(self.transcriptions)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.transcriptions_onehot[idx]), torch.tensor(
            self.strokes_array_list[idx]
        )
