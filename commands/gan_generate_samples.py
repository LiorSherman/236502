import os
import argparse
from gan.utils import generate_samples
from pathlib import Path

def desc():
    return 'Generating samples from generator'


def execute():
    parser = argparse.ArgumentParser(description='GAN : Generate Samples', prog=__name__)
    parser.add_argument('input', help='path to the generator.pt file')
    parser.add_argument('--output', default=None, help='path of the output dir')
    parser.add_argument('--num', type=int, default=1, help='num of samples to generate')
    parser.add_argument('--name', default='My_Track', help='sample names')
    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.input):
        raise ValueError(f"{args.input} is not a valid model path")

    out_path = Path(args.input).parent.absolute()
    if args.output is not None:
        out_path = args.output
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    generate_samples(args.input, out_path, num_samples=args.num, sample_name=args.name)

