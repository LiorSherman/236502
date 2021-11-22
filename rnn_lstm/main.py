import torch
from torch import nn, optim
from model import LSTM
import torch.nn
from preprocess import PreProcessor
from dataset import MusicDataset, split_train_validation
import numpy as np
from torch.utils.data import DataLoader
from training import Trainer
from melody_generation import MelodyGenerator
from rnn_lstm.settings import *

TRAIN = True


def main():
    if TRAIN:
        # pre processing the dataset
        pre = PreProcessor(DEFAULT_DATASET, DEFAULT_PROCESSED_DIR)
        songs = pre.process()

        # splitting the data to train and validation sets
        train_ids, validation_ids = split_train_validation(np.arange(len(songs) - pre.seq_len))

        training_set = MusicDataset(train_ids, songs, pre.mapping_file)
        train_loader = DataLoader(training_set, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)

        validation_set = MusicDataset(validation_ids, songs, pre.mapping_file)
        validation_loader = DataLoader(validation_set, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = LSTM(training_set.vocabulary_size, DEFAULT_HIDDEN_SIZE, training_set.vocabulary_size, pre.mapping_file,
                     device=device)

        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters())

        trainer = Trainer(model, optimiser, criterion, device=device)
        trainer.fit(train_loader, validation_loader, epoches=50)

    gen = MelodyGenerator(DEFAULT_TRAINED_MODELS_DIR)
    song = gen.generate_melody(500, DEFAULT_SEQ_LEN, 0.85)
    gen.save_melody(song)
    print(song)


if __name__ == "__main__":
    main()
