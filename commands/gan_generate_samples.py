import os
import argparse
from gan.utils import generate_samples


def desc():
    return 'Generating samples from generator'


def execute():
    parser = argparse.ArgumentParser(description='GAN : Generate Samples', prog=__name__)
    parser.add_argument('input', help='path to the generator.pt folder')
    parser.add_argument('output', help='path of the output dir')
    parser.add_argument('--num', type=int, default=1, help='num of samples to generate')
    parser.add_argument('--name', default='My Track', help='sample names')
    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid dataset dir path")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    generate_samples(args.input, args.output, num_samples=args.num, sample_name=args.name)

