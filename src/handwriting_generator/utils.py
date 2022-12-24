import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

__all__ = [
    "plotlosses",
    "plot_strokes",
    "plotwindow",
    "load_line_strokes",
    "load_transcriptions",
    "convert_stroke_set_to_array",
    "filter_line_strokes_and_transcriptions",
    "collate_fn",
]


logger = logging.getLogger(__name__)


def load_line_strokes(
    line_strokes_dir: Path, *, show_progress: bool = False
) -> dict[str, list[list[tuple[int, int]]]]:
    result = dict()
    if show_progress:
        iterator = tqdm(line_strokes_dir.rglob("*.xml"))
    else:
        iterator = line_strokes_dir.rglob("*.xml")
    for file in iterator:
        root = ET.parse(file).getroot()
        x_offset = min([int(root[0][i].attrib["x"]) for i in range(1, 4)])
        y_offset = min([int(root[0][i].attrib["y"]) for i in range(1, 4)])
        strokes = []
        for stroke in root[1].findall("Stroke"):
            points = []
            for point in stroke.findall("Point"):
                points.append(
                    (
                        int(point.attrib["x"]) - x_offset,
                        int(point.attrib["y"]) - y_offset,
                    )
                )
            strokes.append(points)
        result[file.stem] = strokes
    return result


def load_transcriptions(
    ascii_dir: Path, *, show_progress: bool = False
) -> dict[str, str]:
    result = dict()
    if show_progress:
        iterator = tqdm(ascii_dir.rglob("*.txt"))
    else:
        iterator = ascii_dir.rglob("*.txt")
    for file in iterator:
        with file.open() as f:
            transcription = f.read()
        if "please enter transcription here" in transcription.lower():
            logger.info(f"Skipping file {file}")
            continue
        transcription = transcription[transcription.find("CSR:") + 6 :]
        transcription_lines = []
        for line in transcription.split("\n"):
            line = line.strip()
            if len(line) < 2:
                continue
            transcription_lines.append(line)

        for i, line in enumerate(transcription_lines, start=1):
            result[file.stem + f"-{str(i).zfill(2)}"] = line
    return result


def filter_line_strokes_and_transcriptions(
    line_strokes: dict[str, list[list[tuple[int, int]]]],
    transcriptions: dict[str, str],
    *,
    show_progress: bool = False,
):
    """Removes line strokes withouts transcriptions and vice-versa"""
    filename_differences = set(transcriptions.keys()).symmetric_difference(
        set(line_strokes.keys())
    )

    if show_progress:
        iterator = tqdm(filename_differences)
    else:
        iterator = filename_differences

    for filename in iterator:
        transcriptions.pop(filename, None)
        line_strokes.pop(filename, None)
    return line_strokes, transcriptions


def convert_stroke_set_to_array(stroke_set: list[list[tuple[int, int]]]) -> np.ndarray:
    n_point = sum(map(len, stroke_set))
    strokes_array = np.zeros((n_point, 3))

    prev_x = 0
    prev_y = 0

    counter = 0

    for strokes in stroke_set:
        for i, point in enumerate(strokes):
            x, y = int(point[0]), int(point[1])
            # Compute the relative distance between current and previous point
            strokes_array[counter, 0] = x - prev_x
            strokes_array[counter, 1] = y - prev_y
            # Store current coordinates for use in next iteration
            prev_x, prev_y = x, y
            # end of stroke
            if i == (len(strokes) - 1):
                strokes_array[counter, 2] = 1
            else:
                strokes_array[counter, 2] = 0
            counter += 1

    # Insert the point (0, 0, 1) at the beginning
    strokes_array = np.insert(strokes_array, 0, [0, 0, 1], axis=0)
    return strokes_array


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    onehot, strokes = zip(*batch)
    onehot_lengths = [len(x) for x in onehot]
    strokes_lens = [len(x) for x in strokes]

    onehot_pad = pad_sequence(onehot, batch_first=True, padding_value=1).float()
    strokes_pad = pad_sequence(strokes, batch_first=True, padding_value=-1).float()

    return onehot_pad, strokes_pad, onehot_lengths, strokes_lens


def plotlosses(losses, title="", xlabel="", ylabel=""):
    epochs = np.arange(losses.size) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_strokes(
    strokes_array: np.ndarray,
    *,
    transcription: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    # Cumulative sum, because they are represented as relative displacement
    x = np.cumsum(strokes_array[:, 0])
    y = np.cumsum(strokes_array[:, 1])
    end_of_stroke = strokes_array[:, 2]
    end_of_stroke_indices, *_ = np.nonzero(end_of_stroke)
    end_of_stroke_indices = np.insert(end_of_stroke_indices, 0, 0)
    idx = 0
    while idx < len(end_of_stroke_indices):
        start_index = end_of_stroke_indices[idx] + 1
        try:
            end_index = end_of_stroke_indices[idx + 1]
        except IndexError:
            end_index = len(x)
        ax.plot(
            x[start_index:end_index],
            y[start_index:end_index],
            linewidth=2.0,
        )
        idx += 1
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    if transcription is not None:
        ax.set_title(transcription)
    return ax


def plotwindow(phis, windows):
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.title("Phis", fontsize=20)
    plt.xlabel("Ascii #", fontsize=15)
    plt.ylabel("Time steps", fontsize=15)
    plt.imshow(phis, interpolation="nearest", aspect="auto", cmap=cm.jet)
    plt.subplot(122)
    plt.title("Soft attention window", fontsize=20)
    plt.xlabel("One-hot vector", fontsize=15)
    plt.ylabel("Time steps", fontsize=15)
    plt.imshow(windows, interpolation="nearest", aspect="auto", cmap=cm.jet)
    plt.show()
