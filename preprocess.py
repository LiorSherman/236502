import music21 as m21
import os
import shutil
import json
import numpy as np
import tensorflow.keras as keras


def transpose(song):
    """Transposes song to C maj/A min
    :param song: Piece to transpose
    :return transposed_song (m21 stream):
    """
    # estimate key using music21
    key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    else:  # the key mode is "minor"
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song


def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:
        ["r", "_", "60", "_", "_", "_", "72" "_"]
    :param song : Piece to encode (m21 stream)
    :param time_step: Duration of each time step in quarter length
    :return:
    """
    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        else:  # symbol is rest
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            # if it's the first time we see a note/rest, let's encode it.
            # Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


class PreProcessor:
    def __init__(self, dataset_path, processed_path, mapping_file, one_string_dataset_file,
                 seq_len, acceptable_durations):
        self.dataset_path = dataset_path
        self.processed_path = processed_path
        self.mapping_file = mapping_file
        self.one_string_dataset_file = one_string_dataset_file
        self.seq_len = seq_len
        self.durations = acceptable_durations
        self.songs = []
        self.outputs = None

    def load(self):
        """Loads all kern pieces in dataset using music21.
        :return songs (list of m21 streams): List containing all pieces
        """
        print(f"Loading songs from \"{self.dataset_path}\"...")
        # resetting the songs of the preprocessor
        self.songs = []
        # go through all the files in dataset and load them with music21
        for path, subdirs, files in os.walk(self.dataset_path):
            for file in files:
                # consider only kern files
                if file[-3:] == "krn":
                    song = m21.converter.parse(os.path.join(path, file))
                    self.songs.append(song)
        print(f"Loaded {len(self.songs)} songs.")

    def _has_acceptable_durations(self, song):
        """Boolean routine that returns True if piece has all acceptable duration, False otherwise.
        :param song: song (m21 stream)
        :return (bool):
        """
        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in self.durations:
                return False
        return True

    def remove_non_acceptable_songs(self):
        """filters the songs without acceptable durations of notes and rests
        """
        print("Removing non-acceptable notes & rests duration songs ...")
        acceptable_songs = []
        for song in self.songs:
            if self._has_acceptable_durations(song):
                acceptable_songs.append(song)
        removed = len(self.songs) - len(acceptable_songs)
        print(f"Removed {removed} songs.")
        self.songs = acceptable_songs

    def transpose_songs(self):
        """
        transposes all songs in the preprocesser.
        stores the transposed songs inside self.songs.
        """
        print("Transposing songs to C maj / A min key...")
        transposed_songs = []
        for song in self.songs:
            transposed_songs.append(transpose(song))
        self.songs = transposed_songs

    def encode_and_save(self):
        """
        encodes the song files into time-series-like music representation.
        stores the new files inside the processed path.
        """
        print("Encoding the dataset into time series representation files...")
        # create a new directory for the processed files
        if os.path.exists(self.processed_path):
            try:
                shutil.rmtree(self.processed_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

        for i, song in enumerate(self.songs):
            # save songs to text file
            save_path = os.path.join(self.processed_path, f"song_{i}")
            with open(save_path, "w") as fp:
                fp.write(encode_song(song))

    def create_single_file_dataset(self):
        """Generates a file collating all the encoded songs and adding new piece delimiters and a JSON mapping of
        all symbols.
        :return songs (str): String containing all songs in dataset + delimiters
        """
        print("Generating single file dataset & unique mapping...")
        new_song_delimiter = "/ " * self.seq_len
        songs = ""

        # load encoded songs and add delimiters
        for path, _, files in os.walk(self.processed_path):
            for file in files:
                file_path = os.path.join(path, file)
                with open(file_path, "r") as fp:
                    song = fp.read()
                songs = songs + song + " " + new_song_delimiter

        # remove empty space from last character of string
        songs = songs[:-1]

        # save string that contains all the dataset
        with open(self.one_string_dataset_file, "w") as fp:
            fp.write(songs)

        print(f"{self.one_string_dataset_file} - single dataset file created!")

        mappings = {}

        # identify the vocabulary
        songs = songs.split()
        vocabulary = list(set(songs))
        self.outputs = len(vocabulary)

        # create mappings
        for i, symbol in enumerate(vocabulary):
            mappings[symbol] = i

        # save vocabulary to a json file
        with open(self.mapping_file, "w") as fp:
            json.dump(mappings, fp, indent=4)
        print(f"{self.mapping_file} - JSON mapping file created!")

    def convert_songs_to_int(self, songs):
        int_songs = []

        # load mappings
        with open(self.mapping_file, "r") as fp:
            mappings = json.load(fp)

        # transform songs string to list
        songs = songs.split()

        # map songs to int
        for symbol in songs:
            int_songs.append(mappings[symbol])

        return int_songs

    def generate_training_sequences(self):
        """Create input and output data samples for training. Each sample is a sequence.
        :return inputs (ndarray): Training inputs
        :return targets (ndarray): Training targets
        """

        # load songs and map them to int
        with open(self.one_string_dataset_file, "r") as fp:
            songs = fp.read()
        int_songs = self.convert_songs_to_int(songs)

        inputs = []
        targets = []

        # generate the training sequences
        num_sequences = len(int_songs) - self.seq_len
        for i in range(num_sequences):
            inputs.append(int_songs[i:i + self.seq_len])
            targets.append(int_songs[i + self.seq_len])

        # one-hot encode the sequences
        vocabulary_size = len(set(int_songs))
        # inputs size: (# of sequences, sequence length, vocabulary size)
        inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
        targets = np.array(targets)

        return inputs, targets

    def process(self, remove_non_acceptable_durations=False, transpose=False):
        """
        Process the songs from the dataset path and save the processed song's representation
        into the processed songs path.
        :param remove_non_acceptable_durations: enable to remove non-acceptable durations of notes
        :param transpose: enable to transpose songs to C maj/ A min
        :return trainx, trainy values.
        """
        # load songs from dataset
        self.load()

        # if flag is set to True - removing songs with non-acceptable durations of notes and rests
        if remove_non_acceptable_durations:
            self.remove_non_acceptable_songs()

        # if flag is set to True - transposing every song to C maj/A min keys
        if transpose:
            self.transpose_songs()

        # encode songs and save the processed files to the processed dataset path
        self.encode_and_save()

        # create a single file dataset combining all songs together and generate a mapping from  each symbol
        # to a integer
        self.create_single_file_dataset()

        # generate training and target sequences for the model from the one file dataset we created
        return self.generate_training_sequences()
