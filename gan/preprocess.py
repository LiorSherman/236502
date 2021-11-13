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


def merge_songs(songs: list, merge_save_dir):
    print('merging')
    merged_songs = []
    for song in tqdm(songs):
        try:
            merged_song = merge_tracks(song)
            if merged_song != None:
                merged_songs.append(merged_song)
                save_path = os.path.join(merge_save_dir, f"merged_{merged_song.name}")
                pypi.write(save_path, merged_song)
        except Exception as e:
            print(e)
            continue
    print(f'total {len(merged_songs)} songs were saved to {merge_save_dir}')
    return merged_songs


def load_multitacks(path):
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
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    midi_songs = load_songs(input_path)  # midi_songs = list[MidiFile]
    # filter songs to be only with 4/4 signature
    filtered_song_names = filter_songs(midi_songs)  # = list[midi path]
    # merge songs into merged midi files
    merged = merge_songs(filtered_song_names, out_path)  # = list[pypi.Multitrack]
    return merged



def preprocess_dataset(input_path, out_path):
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




