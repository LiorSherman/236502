import os
import argparse
from rnn_lstm.melody_generation import MelodyGenerator
from music21 import instrument


def desc():
    return 'Generates melodies from a previously trained model'


def execute():
    parser = argparse.ArgumentParser(description='Rnn melodies generation', prog=__name__)
    parser.add_argument('model', help='path of the trained model')
    parser.add_argument('output', help='path of the output dir')
    parser.add_argument("--num", default=1, type=int, help='number of melodies to generate [default=1]')
    parser.add_argument('--instruments', action='store_true',
                        help="using default drums, piano, bass, strings instruments")
    args = parser.parse_args()

    # check for args validity
    model_path = os.path.join(args.model, 'model.pt')
    mapping_path = os.path.join(args.model, 'mapping.json')
    if not os.path.exists(model_path):
        raise ValueError(f"model.pt file not found in {args.model}")

    if not os.path.exists(mapping_path):
        raise ValueError(f"mapping.json file not found in {args.model}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    instruments = None if not args.instruments else [instrument.UnpitchedPercussion(), instrument.Piano(),
                                                     instrument.ElectricBass(), instrument.StringInstrument()]

    gen = MelodyGenerator(args.model)
    for i in range(1, args.num + 1):
        song = gen.generate_melody(500, 64, 0.85)
        gen.save_melody(song, file_name=os.path.join(args.output, f"melody_{i}.mid"), instruments=instruments)
