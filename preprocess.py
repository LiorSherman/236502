import music21 as m21
import os
import shutil
from typing import List
import json
import numpy as np
from tqdm import tqdm


def pad_song(song: str, max_parts: int) -> str:
    """
    Return the song padded with extra "r" symbols to fit the maximum number of parts
    :param song: string representation of a song
    :param max_parts: The maximum number of parts of all songs
    :return: a padded song with number of parts as the maximum number of parts
    """
    song = song.split(" ")
    padded_song = [pad_with_rests(event, max_parts) for event in song]
    song = " ".join(map(str, padded_song))
    return song


def number_of_parts(event: str) -> int:
    """
    counts the number of parts in a symbol
    :param event: string representation of a symbol
    :return: number of parts in the event
    """
    parts = event.split(',')
    return len(parts)


def pad_with_rests(event: str, max_parts: int) -> str:
    """
    adds "r" to an event in order to fit the maximum parts in all of songs
    :param event: string representation of a symbol
    :param max_parts: number of maximum parts in all songs
    :return: the padded event
    """
    padded_event = event
    parts = event.split(',')
    for i in range(max_parts - len(parts)):
        padded_event += ',r'
    return padded_event


def transpose(song: m21.stream.Score) -> m21.stream.Score:
    """Transposes song to C maj/A min
    :param song: Piece to transpose
    :return transposed_song: the transposed song
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


def encode_song(song: m21.stream.Score, time_step=0.25) -> str:
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:
        ["r", "_", "60", "_", "_", "_", "72" "_"]
    :param song : Piece to encode
    :param time_step: Duration of each time step in quarter length
    :return: a string representation of the song
    """
    encoded_song = []
    encoded_parts = []
    for part in song.parts.stream():
        encoded_part = []
        for event in part.flat.notesAndRests:
            # handle notes
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi  # 60
            # handle rests
            elif isinstance(event, m21.chord.Chord):  # symbol is chord
                symbol = ""
                for pitch in event.pitches:
                    symbol += f".{pitch.midi}"
                symbol = symbol[1:]
            else:
                symbol = "r"

            # convert the note/rest into time series notation
            steps = int(event.duration.quarterLength / time_step)
            for step in range(steps):
                # if it's the first time we see a note/rest, let's encode it.
                # Otherwise, it means we're carrying the same
                # symbol in a new time step
                if step == 0:
                    encoded_part.append(symbol)
                else:
                    encoded_part.append("_")

        encoded_parts.append(encoded_part)
    for tup in zip(*encoded_parts):
        symbol = ""
        for val in tup:
            symbol += f",{val}"
        symbol = symbol[1:]
        encoded_song += [symbol]
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


class PreProcessor:
    """
    a class to help pre process midi and krn files as datasets.
    """
    def __init__(self, dataset_path, output_dir, seq_len=64):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.mapping_file = os.path.join(self.output_dir, 'mapping.json')
        self.seq_len = seq_len
        self.durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
        self.songs = []
        self.encoded_songs = []
        self.one_string_songs = None
        self.outputs = None

    def load(self):
        """
        Loads all kern pieces in dataset using music21.
        """
        # resetting the songs of the preprocessor
        self.songs = []
        # go through all the files in dataset and load them with music21
        num_of_files = 0
        types = ['mid', 'krn']
        for path, subdirs, files in os.walk(self.dataset_path):
            for file in files:
                if file[-3:] in types:
                    num_of_files += 1

        with tqdm(total=num_of_files, desc='loading files') as progress_bar:
            for path, subdirs, files in os.walk(self.dataset_path):
                for file in files:
                    # consider only kern files
                    if file[-3:] in types:
                        progress_bar.set_postfix(file=file)
                        song = m21.converter.parse(os.path.join(path, file))

                        self.songs.append(song)
                        progress_bar.update(1)
            print(f"Loaded {len(self.songs)} songs.")

    def _has_acceptable_durations(self, song: m21.stream.Score) -> bool:
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
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.encoded_songs = [encode_song(song) for song in self.songs]

    def create_single_file_dataset(self):
        """Generates a file collating all the encoded songs and adding new piece delimiters and a JSON mapping of
        all symbols.
        """
        print("Generating single file dataset & unique mapping...")
        new_song_delimiter = "/ " * self.seq_len
        songs = ""

        # load encoded songs and add delimiters
        for song in self.encoded_songs:
            songs = songs + song + " " + new_song_delimiter

        # remove empty space from last character of string
        songs = songs[:-1]
        self.one_string_songs = songs

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

    def convert_songs_to_int(self, songs: str) -> List[int]:
        """
        converts songs from symbol representation into int representation using the JSON map
        :param songs: a string representation of all songs together
        :return: a list of int representation of all songs together
        """
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

    def generate_encoded_dataset(self):
        """
        creating and saving a ndarray of all songs
        :return: ndarray of all songs
        """
        int_songs = self.convert_songs_to_int(self.one_string_songs)
        all_songs_int = np.array(int_songs)
        np.save(os.path.join(self.output_dir, 'dataset.npy'), all_songs_int)
        return all_songs_int

    def normalize_encoded_songs(self):
        # getting the maximum parts of all songs
        max_parts = 0
        for song in self.encoded_songs:
            song = song.split(" ")
            parts = [number_of_parts(event) for event in song]
            max_parts = max(max_parts, max(parts))

        # normalizing all songs to have maximum number of parts
        self.encoded_songs = [pad_song(song, max_parts) for song in self.encoded_songs]

    def process(self, remove_non_acceptable_durations=False, transpose=False):
        """
        Process the songs from the dataset path and save the processed song's representation
        into the processed songs path.
        :param remove_non_acceptable_durations: enable to remove non-acceptable durations of notes
        :param transpose: enable to transpose songs to C maj/ A min
        :return an ndarray of the processed dataset
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

        # normalize songs parts to be as the max part size
        self.normalize_encoded_songs()

        # create a single file dataset combining all songs together and generate a mapping from  each symbol
        # to a integer
        self.create_single_file_dataset()

        # generate training and target sequences for the model from the one file dataset we created
        return self.generate_encoded_dataset()
