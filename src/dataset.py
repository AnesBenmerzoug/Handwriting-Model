import os
import string
import torch
from torch.utils.data import Dataset


class IAMDataset(Dataset):
    """ IAM On-Line Handwriting Dataset Class """

    def __init__(self, parameters, setType='Train'):
        """
        Args:
            parameters (namedTuple): an object containing the session parameters
            setType (string): denotes the type of the dataset
        """
        self.params = parameters
        self.setType = setType.lower()
        self.data_path_list = []
        self.create_data_path_list()

    def create_data_path_list(self):
        type_filename = 'trainset.txt' if self.setType == 'train' else \
                        'testset_v.txt' if self.setType == 'validate' else \
                        'testset_f.txt'
        for filename in os.listdir(os.path.join(self.params.DatasetDir, 'task1')):
            if filename == type_filename:
                with open(os.path.join(self.params.DatasetDir, 'task1', filename)) as setTextFile:
                    for line in setTextFile:
                        ascii_path = os.path.join(self.params.DatasetDir, 'ascii')
                        stroke_path = os.path.join(self.params.DatasetDir, 'lineStrokes')
                        line = line[1:-1]  # Skip the initial space and the last new line character
                        dashIndex = line.find('-')
                        ascii_path = os.path.join(ascii_path, line[:dashIndex])
                        stroke_path = os.path.join(stroke_path, line[:dashIndex])
                        if line[-1] in string.ascii_lowercase:
                            ascii_path = os.path.join(ascii_path, line[:-1])
                            stroke_path = os.path.join(stroke_path, line[:-1])
                        else:
                            ascii_path = os.path.join(ascii_path, line)
                            stroke_path = os.path.join(stroke_path, line)
                        ascii_path = os.path.join(ascii_path, line + '.txt')
                        strokes = []
                        for stroke in os.listdir(stroke_path):
                            if line in stroke:
                                strokes.append(os.path.join(stroke_path, stroke))
                        strokes.sort(key=lambda name: int(name[-6:-4]))
                        self.data_path_list.append((ascii_path, strokes))

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        ascii_path, strokes_paths = self.data_path_list[idx]
        pass
