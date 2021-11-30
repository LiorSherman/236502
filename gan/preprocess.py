import os
import pathlib
from tqdm import tqdm
from mido import MidiFile
import pypianoroll as pypi
from copy import deepcopy
from gan.settings import BEAT_RESOLUTION, N_TRACKS, N_BARS, N_STEPS_PER_BAR
import numpy as np
# Initialize an empty list to collect the results
results = []
BASEDIR = pathlib.Path(__file__).parent.resolve()
current_folder = os.path.expanduser(BASEDIR)
#merge_dir = os.path.join(current_folder, 'merged')
merge_out_dir = os.path.join('gan', 'merged')

def load_songs(path):
    """ Loads midi songs from path
    :param path: path to midi dir
    :return: List of MidiFiles
    """
    print(f"Loading songs from \"{path}\"...")
    # resetting the songs of the preprocessor
    songs = []
    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(path):
        for file in files:
            # consider only mid files
            if file[-3:] == "mid":
                try:
                    song = MidiFile(os.path.join(path, file))
                    songs.append(song)
                    print('.', end="")
                except Exception as e:
                    print()
                    print(e)
                    continue
    print(f"Loaded {len(songs)} songs.")
    return songs


def filter_songs(midi_songs: list):
    """ Filters out midis which doesnt have 4/4 time signature
    :param midi_songs: list of Midifiles
    :return: List of paths to midi files containing 4/4 time signature
    """
    def is_ok(midi):
        for track in midi.tracks:
            for msg in track:
                msgInfo = vars(msg)
                if msgInfo["type"] == "time_signature":
                    try:
                        if msgInfo["numerator"] != 4 or msgInfo["denominator"] != 4:
                            return False
                    except Exception as e:
                        print(f"Error occured in {midi}")
                        print(e)
        return True

    print('filtering songs')
    filtered_song_names = [midi.filename for midi in tqdm(midi_songs) if is_ok(midi)]
    print(f'{len(filtered_song_names)}/{len(midi_songs)} songs are 4/4')
    return filtered_song_names


def merge_tracks(song_path):
    """ Merges a multi track midi file into 4 tracks - drums, piano, bass, ensemble
    All similiar instrument tracks are combined together.
    Notice: returns none incase at least one of the above tracks is absent
    :param song_path: path to midi file
    :return: pypi.Multitrack containing 4 tracks as described above, or None in case at least one track is absent.
    """
    multitrack = pypi.read(song_path, resolution=BEAT_RESOLUTION)
    multitrack.name = os.path.basename(song_path)

    midi_instrument_programs = {"Drums": 0, "Piano": 0, "Bass": 33, "Ensemble": 48, }
    track_list_per_instrument = {instrument: [] for instrument in midi_instrument_programs}

    for i, track in enumerate(multitrack.tracks):
        if track.is_drum:
            track_list_per_instrument["Drums"].append(i)
        elif 0 <= track.program < 8:
            track_list_per_instrument["Piano"].append(i)
        elif 32 <= track.program < 40:
            track_list_per_instrument["Bass"].append(i)
        elif 48 <= track.program < 56:
            track_list_per_instrument["Ensemble"].append(i)

    # if not all instrument found return None
    for instrument in track_list_per_instrument.keys():
        if len(track_list_per_instrument[instrument]) < 1:
            return None

    merged_tracks = []
    for i, instrument in enumerate(track_list_per_instrument.keys()):

        tracks_to_merge = []
        for track_idx in track_list_per_instrument[instrument]:
            tracks_to_merge.append(multitrack.tracks[track_idx])
        auxM = pypi.Multitrack(tracks=tracks_to_merge, tempo=multitrack.tempo, downbeat=multitrack.downbeat,
                                      resolution=multitrack.resolution)
        merged = auxM.blend("max")
        merged_tracks.append(pypi.Track(pianoroll=merged, program=midi_instrument_programs[instrument], is_drum=(i == 0),
                                        name=instrument).standardize().binarize())

    m = pypi.Multitrack(tracks=merged_tracks, tempo=multitrack.tempo[0], downbeat=multitrack.downbeat,
                               resolution=multitrack.resolution, name=multitrack.name)
    return m


def merge_songs(l_midis_paths: list, merge_save_dir):
    """ Merges multi track midi files into 4 tracks - drums, piano, bass, ensemble,
    and saves each midi file name 'some_midi.mid' to an output folder under the name 'merged_some_midi.mid'
    Calls'merge_tracks()' function. See documentaion of 'merge_tracks().
    :param l_midis_paths: List of midi paths
    :param merge_save_dir: dir output path for saving merged midis
    :return: List of pypi.Multitrack containing 4 tracks as described above, or None in case at least one track is absent.
    """
    print('merging')
    merged_songs = []
    for midi_path in tqdm(l_midis_paths):
        try:
            merged_song = merge_tracks(midi_path)
            if merged_song is not None:
                merged_songs.append(merged_song)
                save_path = os.path.join(merge_save_dir, f"merged_{merged_song.name}")
                pypi.write(save_path, merged_song)
        except Exception as e:
            print(e)
            continue
    print(f'total {len(merged_songs)} songs were saved to {merge_save_dir}')
    return merged_songs


def load_multitacks(path):
    """ Loads multitracks from path
    :param path: path to midi directory
    :return: List of pypi.Multitracks
    """
    multitracks = []
    for path, subdirs, files in os.walk(path):
        for file in files:
            try:
                m = pypi.read(os.path.join(path,file), resolution=BEAT_RESOLUTION)
                multitracks.append(m)
            except Exception as e:
                print(f"Error occured in {file}")
                print(e)
    return multitracks


def prepare_dataset(input_path, out_path=merge_out_dir):
    """ Prepares midi dataset for preprocessing.
    1 - Loads midis from path
    2 - Filters out songs not containing 4/4 time signature
    3 - Merges x-track midis into 4-track midis
    See envoked function documentations for more info.
    :param input_path: : path to midi dir
    :param out_path: dir output path for saving merged midis
    :return: List of pypi.Multitrack containing 4 tracks as described above, or None in case at least one track is absent.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    midi_songs = load_songs(input_path)  # midi_songs = list[MidiFile]
    # filter songs to be only with 4/4 signature
    filtered_song_names = filter_songs(midi_songs)  # = list[midi path]
    # merge songs into merged midi files
    merged = merge_songs(filtered_song_names, out_path)  # = list[pypi.Multitrack]
    return merged



def preprocess_dataset(input_path, out_path):
    """ Preprocess midi dataset for the data loader before training
    1 - Loads 4-track midis from input_path
    2 - Extracts all tracks pianorolls, binarizes, and stacks them together.
    3 - Filters out low ([0:23]) and high ([109:128]) pitches
    4 - Splits pianorolls into phrases(4 bars)
    5 - Filteres out "quite" samples, while "quite" sample is a phrase containing at least one bar at any of the tracks
        that is silent.
    :param input_path: path to prepared midi dir containing 4-track midis
    :param out_path: dir output path for saving dataset.npy file
    :return: np.ndarray, shape=(-1, n_tracks=4, n_bars=4, n_steps_per_bar=16, n_pitches=84)
            The stacked pianoroll.
    """
    if not os.path.exists(pathlib.Path(out_path).parent.resolve()):
        os.makedirs(pathlib.Path(out_path).parent.resolve())

    def get_stacked_pianoroll(multitrack):
        """
        Return a stacked multitrack pianoroll. The shape of the return array is
        (n_time_steps, 128, n_tracks).

        Returns
        -------
        stacked : np.ndarray, shape=(n_time_steps, 128, n_tracks)
            The stacked pianoroll.

        """
        multitrack = deepcopy(multitrack)
        multitrack.pad_to_same()
        stacked = np.stack([track.pianoroll for track in multitrack.tracks], -1)
        return stacked

    multitracks = load_multitacks(input_path)

    for multitrack in tqdm(multitracks):
        if len(multitrack.tracks) != N_TRACKS:
            continue

        # Pad to multtple
        multitrack.pad_to_multiple(N_BARS * N_STEPS_PER_BAR)

        # Binarize the pianoroll
        multitrack.binarize()

        # Sort the tracks according to program number
        multitrack.tracks.sort(key=lambda x: x.program)

        # Bring the drum track to the first track
        multitrack.tracks.sort(key=lambda x: ~x.is_drum)

        # Get the stacked pianoroll
        pianoroll = get_stacked_pianoroll(multitrack)

        # Check length
        if pianoroll.shape[0] < 4 * 4 * BEAT_RESOLUTION:
            continue

        # Keep only the mid-range pitches
        pianoroll = pianoroll[:, 24:108]  # (ndarray:(n_time_steps, 84, n_tracks))

        pianoroll = pianoroll.reshape(-1, N_BARS, N_STEPS_PER_BAR, 84, N_TRACKS)

        pianoroll1 = np.ones(pianoroll.shape, dtype=np.int8)
        pianoroll1[pianoroll == False] = -1
        results.append(pianoroll1)


    result = np.concatenate(results, 0).transpose([0, 4, 1, 2, 3])  # (n_samples, n_tracks, n_bars, n_steps_per_bar, 84)

    #Filter silent samples - filters out samples containing at least silent bar for any of the tracks
    result = result[np.all(
        np.count_nonzero((result[:, :, :].sum(-1).sum(-1) + 16 * 84)[:], axis=1) >= np.ones((N_BARS),
                                                                                            dtype=np.int) * N_TRACKS,
        axis=1)]
    print(f'Dataset shape : {result.shape}')
    with open(out_path, 'wb') as f:
        np.save(f, result)
        print(f'Dataset saved to {out_path}')
    return result




