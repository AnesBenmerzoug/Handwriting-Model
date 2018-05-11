import torch


class Batchifier(object):
    def __init__(self, parameters):
        self.params = parameters

    def collate_fn(self, batch_list):
        onehot_list, stroke_list = zip(*map(lambda elem: (elem[0], elem[1][:self.params.min_num_points, :]), batch_list))
        max_onehot_len = max([onehot.size(0) for onehot in onehot_list])
        onehot_padded = []
        for onehot in onehot_list:
            zero = torch.zeros(max_onehot_len - onehot.size(0), onehot.size(1))
            onehot = torch.cat((onehot, zero), dim=0)
            onehot_padded.append(onehot)
        return torch.stack(onehot_padded, dim=0), torch.stack(stroke_list, dim=0)
