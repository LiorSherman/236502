import os
import argparse
from rnn_lstm.preprocess import PreProcessor


def desc():
    return 'Processing midi and krn files into a proccesed .npy and mapping.json files'


def execute():
    parser = argparse.ArgumentParser(description='Rnn preprocess', prog=__name__)
    parser.add_argument('input', help='path of the dataset to process')
    parser.add_argument('output', help='path of the output dir')

    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid dataset dir path")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    pre = PreProcessor(args.input, args.output)
    _ = pre.process()

