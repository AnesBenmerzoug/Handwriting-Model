import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.progress import track
from torch.nn.utils.rnn import pad_sequence

__all__ = [
    "plot_strokes",
    "plot_window_weights",
    "load_line_strokes",
    "load_transcriptions",
    "convert_stroke_set_to_array",
    "filter_line_strokes_and_transcriptions",
    "collate_fn",
    "find_best_model_checkpoint",
    "batch_index_select",
]


logger = logging.getLogger(__name__)


def load_line_strokes(
    line_strokes_dir: Path, *, show_progress: bool = False
) -> dict[str, list[list[tuple[int, int]]]]:
    result = dict()
    if show_progress:
        iterator = track(line_strokes_dir.rglob("*.xml"))
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
        iterator = track(ascii_dir.rglob("*.txt"))
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
        iterator = track(filename_differences)
    else:
        iterator = filename_differences

    for filename in iterator:
        transcriptions.pop(filename, None)
        line_strokes.pop(filename, None)
    return line_strokes, transcriptions


def convert_stroke_set_to_array(stroke_set: list[list[tuple[int, int]]]) -> np.ndarray:
    """Converts stroke sets to arrays and at the same time remove the trend from y-axis values.
    The returned arrays contains the first order difference of the x and y values instead of the raw values.
    """
    n_point = sum(map(len, stroke_set))
    strokes_array = np.zeros((n_point, 3))
    counter = -1

    for strokes in stroke_set:
        for i, point in enumerate(strokes):
            counter += 1
            x, y = int(point[0]), int(point[1])
            strokes_array[counter, 0] = x
            strokes_array[counter, 1] = y
            strokes_array[counter, 2] = 0
        # end of stroke
        strokes_array[counter, 2] = 1

    # Remove trend on the y-axis
    X = np.arange(0, len(strokes_array))
    z = np.polyfit(X, strokes_array[:, 1], deg=1)
    y_trend = np.polyval(z, X)
    strokes_array[:, 1] -= y_trend

    # Normalize x of y values
    strokes_array[:, :2] = strokes_array[:, :2] / np.max(strokes_array[:, :2])

    # Replace the x and y values with their respective 1st order difference
    strokes_array[:, :2] = np.diff(strokes_array[:, :2], prepend=0, axis=0)

    # Insert the point (0, 0, 1) at the beginning
    strokes_array = np.insert(strokes_array, 0, [0.0, 0.0, 1.0], axis=0)
    return strokes_array


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[
    torch.Tensor, torch.Tensor, np.ndarray[np.int_], np.ndarray[np.int_], list[str]
]:
    strokes, onehot, transcriptions = zip(*batch)

    strokes_lengths = torch.as_tensor([len(x) for x in strokes], dtype=torch.int64)
    onehot_lengths = torch.as_tensor([len(x) for x in onehot], dtype=torch.int64)

    strokes_pad = pad_sequence(strokes, batch_first=True, padding_value=0).float()
    onehot_pad = pad_sequence(onehot, batch_first=True, padding_value=0).float()

    # Sort all tensors by descending strokes lengths
    strokes_lengths, sorted_indices = torch.sort(strokes_lengths, descending=True)
    strokes_pad = strokes_pad.index_select(0, sorted_indices)
    onehot_pad = onehot_pad.index_select(0, sorted_indices)
    onehot_lengths = onehot_lengths.index_select(0, sorted_indices)
    transcriptions = [transcriptions[i] for i in sorted_indices]

    # Convert length tensors to numpy arrays
    # to avoid errors with pad_packed_sequence and pack_padded_sequence
    strokes_lengths = strokes_lengths.numpy()
    onehot_lengths = onehot_lengths.numpy()

    return strokes_pad, onehot_pad, strokes_lengths, onehot_lengths, transcriptions


def find_best_model_checkpoint(logs_dir: Path) -> Path:
    logs_version_dirs = sorted(list(logs_dir.iterdir()), reverse=True)
    logs_version_dirs = list(
        filter(lambda x: x.joinpath("checkpoints").exists(), logs_version_dirs)
    )
    logs_last_version_dir = logs_version_dirs[0]
    checkpoint_dir = logs_last_version_dir / "checkpoints"
    checkpoint_paths = sorted(
        [x for x in checkpoint_dir.glob("*.ckpt") if "last" not in x.name]
    )
    checkpoint_path = checkpoint_paths[-1]
    return checkpoint_path


def batch_index_select(
    input: torch.Tensor, dim: int, indices: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [torch.index_select(x, dim, idx) for x, idx in zip(input, indices)]
    )


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


def plot_window_weights(
    strokes: np.ndarray, phi: np.ndarray, transcription: str
) -> Figure:
    fig, axes = plt.subplots(2, 1)
    plot = axes[0].imshow(phi.T, interpolation="nearest", aspect="auto", cmap=cm.jet)
    axes[0].set_title("Phis")
    axes[0].set_ylabel(transcription)
    cax = axes[0].inset_axes([1.02, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(plot, ax=axes[0], cax=cax)
    cbar.minorticks_on()

    plot_strokes(strokes, ax=axes[1])
    axes[1].set_axis_off()

    fig.tight_layout()
    return fig
