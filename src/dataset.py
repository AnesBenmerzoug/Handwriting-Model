import logging
import os
import pickle
import string
import tarfile

import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.constants import DATA_DIR

__all__ = ["IAMDataset"]

logger = logging.getLogger(__name__)


class IAMDataset(Dataset):
    """ IAM On-Line Handwriting Dataset Class """

    def __init__(self, parameters):
        """
        Args:
            parameters (namedTuple): an object containing the session parameters
        """
        self.params = parameters
        self.ascii_data_file = DATA_DIR / "ascii-all.tar.gz"
        self.line_strokes_data_file = DATA_DIR / "lineStrokes-all.tar.gz"

        self.ascii_data_dir = DATA_DIR / "ascii"
        self.line_strokes_data_dir = DATA_DIR / "lineStrokes"

        self.data_filename = DATA_DIR / "strokes_data.pickled"

        if not (self.ascii_data_file.exists() and self.line_strokes_data_file.exists()):
            raise FileNotFoundError(
                f"Could not find data files. "
                "Please make sure to follow the instructions in the README "
                "to download and set up the dataset."
            )

        if not self.ascii_data_dir.exists():
            with tarfile.open(self.ascii_file, "r") as tar:
                tar.extractall(path=DATA_DIR)

        if not self.line_strokes_data_dir.exists():
            with tarfile.open(self.line_strokes_file, "r") as tar:
                tar.extractall(path=DATA_DIR)

        # space + uppercase and lowercase letters, their indices will be shifted when transforming them
        # to one hot in order to have index 0 for unknown characters
        self.alphabet = "".join(
            [" "] + [c for c in string.ascii_lowercase + string.ascii_uppercase]
        )
        self.length = 0
        self.limit = 300
        self.min_num_points = self.params.min_num_points

        self.ascii = []
        self.strokes = []
        self.ascii_onehot = []

        if not (os.path.exists(self.data_filename)):
            logger.info("Creating file {}".format(self.data_filename))
            self.prepocess_data()
        else:
            logger.info("File {} exists already".format(self.data_filename))

        self.load_data()

    def prepocess_data(self):
        def create_data_path_list():
            data_path_list = []

            ascii_dir = self.ascii_data_dir

            for root, dirs, files in os.walk(ascii_dir):
                if not files:
                    continue
                for f in files:
                    ascii_path = os.path.join(root, f)
                    strokes_dir = root.replace("ascii", "lineStrokes")
                    stroke_paths = []
                    if os.path.isdir(strokes_dir):
                        for stroke_file in os.listdir(strokes_dir):
                            if f[:-4] in stroke_file:
                                stroke_paths.append(
                                    os.path.join(strokes_dir, stroke_file)
                                )
                        stroke_paths.sort(key=lambda name: int(name[-6:-4]))
                        data_path_list.append((ascii_path, stroke_paths))
            return data_path_list

        def getAscii(filename):
            with open(filename, "r") as f:
                text = f.read()
            text = text[text.find("CSR:") + 6 :]
            return text.split("\n")

        def getStrokes(filename_list):
            result = []
            for stroke_file in filename_list:
                root = ET.parse(stroke_file).getroot()
                x_offset = min([float(root[0][i].attrib["x"]) for i in range(1, 4)])
                y_offset = min([float(root[0][i].attrib["y"]) for i in range(1, 4)])
                strokes = []
                for stroke in root[1].findall("Stroke"):
                    points = []
                    for point in stroke.findall("Point"):
                        points.append(
                            (
                                float(point.attrib["x"]) - x_offset,
                                float(point.attrib["y"]) - y_offset,
                            )
                        )
                    strokes.append(points)
                result.append(strokes)
            return result

        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3))

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    # Limit the relative distance between points
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if k == (len(stroke[j]) - 1):  # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        data_path_list = create_data_path_list()
        text_array = []
        strokes_array = []

        with logging_redirect_tqdm():
            for ascii_file, strokes_files in tqdm(data_path_list):
                # Get the text from the files
                text_list = getAscii(ascii_file)

                # Get the strokes from the files
                strokes_list = getStrokes(strokes_files)

                for text, strokes in zip(text_list, strokes_list):
                    if len(text) > 10:
                        text_array.append(text)
                        strokes_array.append(convert_stroke_to_array(strokes))
                    else:
                        logger.info("Text was too short: {}".format(text))

        assert len(text_array) == len(strokes_array)
        with open(self.data_filename, "wb+") as f:
            pickle.dump([text_array, strokes_array], f)

    def load_data(self):
        with open(self.data_filename, "rb") as f:
            raw_ascii, raw_strokes = pickle.load(f)
        self.ascii_onehot = []
        for sentence, stroke in zip(raw_ascii, raw_strokes):
            if len(stroke) <= self.min_num_points:
                continue
            else:
                stroke = stroke[: self.min_num_points, :]
            self.ascii.append(sentence)
            # Insert the point (0, 0, 1) at the beginning
            stroke = np.insert(stroke, 0, [0.0, 0.0, 1.0], axis=0)
            self.strokes.append(stroke)
            # Since we removed some points from the strokes, we should limit the size of the sentence too
            sentence_size = int(self.min_num_points / 22)
            if len(sentence) >= sentence_size:
                sentence = sentence[:sentence_size]
            else:
                sentence = sentence + "_" * (sentence_size - len(sentence))
            onehot = np.zeros(
                shape=(len(sentence), len(self.alphabet) + 1), dtype=np.uint8
            )
            indices = [self.alphabet.find(c) + 1 for c in sentence]
            onehot[np.arange(len(sentence)), indices] = 1
            self.ascii_onehot.append(onehot)
        self.strokes = np.stack(self.strokes, axis=0)
        # Scale the X and Y components down by dividing by their standard deviation
        # self.strokes[:, 1:, :2] = self.strokes[:, 1:, :2] - np.mean(self.strokes[:, 1:, :2], axis=(0, 1))
        self.strokes[:, 1:, :2] = self.strokes[:, 1:, :2] / np.std(
            self.strokes[:, 1:, :2], axis=(0, 1)
        )
        self.length = len(self.ascii)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx (integer): index of the element to get
        :return: onehot encoded ascii string Tensor, strokes Tensor
        """
        return torch.Tensor(self.ascii_onehot[idx]), torch.Tensor(self.strokes[idx])
