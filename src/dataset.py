from __future__ import print_function
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import string
import os


class IAMDataset(Dataset):
    """ IAM On-Line Handwriting Dataset Class """

    def __init__(self, parameters, setType='Training'):
        """
        Args:
            parameters (namedTuple): an object containing the session parameters
            setType (string): denotes the type of the dataset. It's possible values are:
                                'Training', 'Validation', 'Testing'
        """
        self.params = parameters
        assert(setType.lower() == 'training' or setType.lower() == 'validation' or setType.lower() == 'testing')
        self.setType = setType.lower()
        self.data_filename = os.path.join(self.params.DatasetDir, self.setType + '_data.pickled')
        # '_' represents unknown characters, characters not in the alphabet
        self.alphabet = ['_', ' ', '-', '.', ',', "'", '"', '!', '?', '(', ')'] + \
                        [c for c in string.digits + string.ascii_uppercase + string.ascii_lowercase]
        self.length = 0

        self.ascii = None
        self.strokes = None
        self.ascii_onehot = None

        if not (os.path.exists(self.data_filename)):
            print("Creating file {}".format(self.data_filename))
            self.prepocess_data()
        else:
            print("File {} exists already".format(self.data_filename))

        self.load_data()

    def get_alphabet(self):
        with open(os.path.join(self.params.DatasetDir, 'task1', 'letters')) as alphabet:
            for letter in alphabet:
                self.alphabet.append(letter.replace('\n', ''))

    def prepocess_data(self):
        data_path_list = self.create_data_path_list()
        text_array = []
        strokes_array = []

        def getAscii(filename):
            with open(filename, "r") as f:
                text = f.read()
            text = text[text.find('CSR:') + 6:]
            return text.split('\n')

        def getStrokes(filename_list):
            result = []
            for stroke_file in filename_list:
                root = ET.parse(stroke_file).getroot()
                x_offset = min([float(root[0][i].attrib['x']) for i in range(1, 4)])
                y_offset = min([float(root[0][i].attrib['y']) for i in range(1, 4)])
                strokes = []
                for stroke in root[1].findall('Stroke'):
                    points = []
                    for point in stroke.findall('Point'):
                        points.append((float(point.attrib['x']) - x_offset,
                                       float(point.attrib['y']) - y_offset))
                    strokes.append(points)
                result.append(strokes)
            return result

        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if k == (len(stroke[j]) - 1):  # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        for ascii_file, strokes_files in data_path_list:
            # Get the text from the files
            text_list = getAscii(ascii_file)

            # Get the strokes from the files
            strokes_list = getStrokes(strokes_files)

            for text, strokes in zip(text_list, strokes_list):
                if len(text) > 10:
                    text_array.append(text)
                    strokes_array.append(convert_stroke_to_array(strokes))
                else:
                    print("\nText was too short: {}".format(text))

        assert (len(text_array) == len(strokes_array))
        with open(self.data_filename, 'wb+') as f:
            pickle.dump([text_array, strokes_array], f)

    def create_data_path_list(self):
        type_filename = 'trainset.txt' if self.setType == 'train' else \
            'testset_v.txt' if self.setType == 'validate' else \
                'testset_f.txt'
        data_path_list = []
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
                        data_path_list.append((ascii_path, strokes))
        return data_path_list

    def load_data(self):
        with open(self.data_filename, 'rb') as f:
            self.ascii, self.strokes = pickle.load(f)
        self.length = len(self.ascii)
        self.ascii_onehot = []
        for sentence in self.ascii:
            onehot = np.zeros(shape=(len(sentence), len(self.alphabet)), dtype=np.uint8).tolist()
            for i, c in enumerate(sentence):
                if c in self.alphabet:
                    onehot[i][self.alphabet.index(c)] = 1
                else:
                    onehot[i][0] = 1
            self.ascii_onehot.append(onehot)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx (integer): index of the element to get
        :return: onehot encoded ascii string Tensor, strokes Tensor
        """
        return torch.LongTensor(self.ascii_onehot[idx]), torch.Tensor(self.strokes[idx])
