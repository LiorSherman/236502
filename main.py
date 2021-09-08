from preprocess import PreProcessor
from hyperparameters import preprocessor_params
from training import Trainer
from rnn_lstm import RnnLstm
from hyperparameters import melody_generator_params
from melody_generation import MelodyGenerator
from tensorflow.keras.backend import manual_variable_initialization

if __name__ == "__main__":
    TRAIN = False

    if TRAIN:
        pre = PreProcessor(**preprocessor_params)
        x, y = pre.process(remove_non_acceptable_durations=False, transpose=False)
        rnn = RnnLstm(pre.outputs, [256], 0.001, pre.mapping_file)
        manual_variable_initialization(True)
        trainer = Trainer(rnn)
        trainer.fit(x, y, 64)
        trainer.save_model("trained_models")
        gen = MelodyGenerator(**melody_generator_params, model=trainer.model)
    else:
        gen = MelodyGenerator(**melody_generator_params)
        gen.load_model("trained_models/top10")
    song = gen.generate_melody(500, 64, 0.8)
    gen.save_melody(song)
