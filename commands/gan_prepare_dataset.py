import os
import argparse
from gan.preprocess import prepare_dataset


def desc():
    return 'Preparing 4-track-midi dataset from midi dataset'


def execute():
    parser = argparse.ArgumentParser(description='GAN : Prepare Dataset', prog=__name__)
    parser.add_argument('input', help='path to the midi folder')
    parser.add_argument('output', help='path of the output dir')

    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid dataset dir path")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    prepare_dataset(args.input, args.output)

