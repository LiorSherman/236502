import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


DEFAULT_DATASET = os.path.join(BASE_DIR, 'rnn_lstm', 'dataset', 'deutschl', 'test')
DEFAULT_EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'rnn_lstm', 'experiments_results')
DEFAULT_SEQ_LEN = 64
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_PROCESSED_DIR = os.path.join(BASE_DIR, 'rnn_lstm', 'processed_dataset')
DEFAULT_TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'rnn_lstm', 'trained_models')

