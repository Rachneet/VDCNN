import numpy as np
import csv
from torch.utils.data import Dataset
import sys
csv.field_size_limit(sys.maxsize)

class MyDataset(Dataset):
    def __init__(self, data_path, max_length=1014):
        self.data_path = data_path
        self.vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")

        # self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path,encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                # if len(line) == 3:
                #     text = "{} {}".format(line[1].lower(), line[2].lower())
                # else:
                #     text = "{}".format(line[1].lower())
                label = int(line[0])
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    # gets the length
    def __len__(self):
        return self.length

    # gets data based on given index
    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = [self.vocabulary.index(i) for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        elif len(data) == 0:
            data += [0] * (self.max_length)
        label = self.labels[index]
        return np.array(data, dtype=np.int64), label