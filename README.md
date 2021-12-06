# Melody generation

[MusicAI](https://github.com/LiorSherman/236502) is a melody generation project using RNN LSTM and GAN models.

## Datasets

- [Esac](http://www.esac-data.org) `Essen Associative Code` including more than 20k .krn files of european folk melodies.
- [Snes](https://www.vgmusic.com/music/console/nintendo/snes/) over 6709 .mid files of `Super Nintendo` melodies.

## Sample results
### RNN LSTM model
- [sample 1](https://github.com/LiorSherman/236502/tree/master/best_results/rnn_piano_1.mp3)
- [sample 2](https://github.com/LiorSherman/236502/tree/master/best_results/rnn_piano_2.mp3)
- [sample 3](https://github.com/LiorSherman/236502/tree/master/best_results/rnn_multi.mp3)

### GAN model
- [sample 1](https://github.com/LiorSherman/236502/tree/master/best_results/gan_1.mp3)
- [sample 2](https://github.com/LiorSherman/236502/tree/master/best_results/gan_2.mp3)
- [sample 3](https://github.com/LiorSherman/236502/tree/master/best_results/gan_3.mp3)


## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

> __Optional (Recommended)__ - installing [Musescore](https://musescore.org/en): free music notation software which will allow to display and play midi files (Not installing might cause some features not to work properly).
- Create your virtual environment
- Install the dependencies using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```
  
### Command Line Interface

> We have created a python command line manager for use on all OS

```sh
$ python manage.py -h
Usage: python manage.py [command_name] [command args] [-h]

Command list: 
        gan_generate_samples: Generating samples from generator
        rnn_process_train_gen: Generating melodies using the WHOLE process of pre processing training and generating
        gan_preprocess_dataset: Preprocessing 4-track-merged midi dataset to npy file
        rnn_train: Training and saving a model using unprocessed dataset
        rnn_preprocess: Processing midi and krn files into a proccesed .npy and mapping.json files
        gan_prepare_dataset: Preparing 4-track-midi dataset from midi dataset
        rnn_generate: Generates melodies from a previously trained model
        rnn_exp: Experimenting with different hyperparameters
        gan_train: Training Gan

Each command help menu can be accessed via -h when using it
```
> Example: processing a dataset, training a RNN LSTM model and generating 10 melodies in `my_demo` directory

```sh
$ python manage.py rnn_process_train_gen rnn_lstm/dataset/deutschl/test my_demo --num 10 --epochs 50
```

### Using pre trained models

You can use pre trained models easily with the CLI.

> Example: generating 5 samples using pre trained GAN model on `my_demo` dir

```sh
$ python manage.py gan_generate_samples gan/pre_trained_models/modelA/generator.pt my_demo --num 5
