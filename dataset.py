import json

import numpy as np
import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, list_ids: np.array, songs, dictionary_path: str, seq_len=64):
        self.list_ids = list_ids
        self.all_songs_int = songs
        with open(dictionary_path, "r") as f:
            self.dictionary = json.load(f)

        self.vocabulary_size = len(self.dictionary)
        self.sequence_length = seq_len
        self.num_sequences = len(self.all_songs_int) - self.sequence_length

    @staticmethod
    def one_hot(input_data: np.array, number_classes: int):
        return np.eye(number_classes)[input_data.reshape(-1)]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        i = self.list_ids[index]

        sequence = self.all_songs_int[i: i + self.sequence_length]
        sequence = self.one_hot(sequence, self.vocabulary_size)
        label = self.all_songs_int[i + self.sequence_length]
        return torch.from_numpy(sequence).float(), label


def split_train_validation(data: np.array, train_size: float = 0.90):
    choice = np.random.choice(
        range(data.shape[0]), size=(int(len(data) * train_size),), replace=False
    )
    train_mask = np.zeros(data.shape[0], dtype=bool)
    train_mask[choice] = True
    validation_mask = ~train_mask

    return data[train_mask], data[validation_mask]
