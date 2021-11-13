import os
import argparse
from gan.preprocess import preprocess_dataset


def desc():
    return 'Preprocessing 4-track-merged midi dataset to npy file'


def execute():
    parser = argparse.ArgumentParser(description='GAN : Preprocess Dataset', prog=__name__)
    parser.add_argument('input', help='path to the 4-track-midi folder')
    parser.add_argument('output', help='path of the output dir')
    parser.add_argument('--name', default='my_dataset', help='npy dataset file name')

    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid dataset dir path")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    preprocess_dataset(args.input, os.path.join(args.output, args.name+'.npy'))

