import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import split_train_validation, MusicDataset
from model import LSTM
import numpy as np
from preprocess import PreProcessor
from training import Trainer


def desc():
    return 'Training and saving a model using unprocessed dataset'


def execute():
    parser = argparse.ArgumentParser(description='Rnn train', prog=__name__)
    parser.add_argument('input', help='path of the dataset to process')
    parser.add_argument('output', help='path of the output dir')

    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid dataset dir path")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # pre processing the dataset
    pre = PreProcessor(args.input, args.output)
    songs = pre.process()

    # splitting the data to train and validation sets
    train_ids, validation_ids = split_train_validation(np.arange(len(songs) - pre.seq_len))

    training_set = MusicDataset(train_ids, songs, pre.mapping_file)
    train_loader = DataLoader(training_set, batch_size=64, shuffle=True)

    validation_set = MusicDataset(validation_ids, songs, pre.mapping_file)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTM(training_set.vocabulary_size, 256, training_set.vocabulary_size, pre.mapping_file, device=device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    trainer = Trainer(model, optimiser, criterion, device=device, output=args.output)
    trainer.fit(train_loader, validation_loader)
