import pathlib
import os
import numpy as np
import pypianoroll as pypi
import torch
from torch import nn
import matplotlib.pyplot as plt
from music21 import converter
BASEDIR = pathlib.Path(__file__).parent.resolve()
current_folder = os.path.expanduser(BASEDIR)
from gan.settings import N_TRACKS, Z_DIM

class WassersteinLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, pred, target):
        loss = - torch.mean(pred * target)
        return loss


class GradientPenalty(nn.Module):
    def __init(self, ):
        super().__init__()

    def forward(self, inputs, outputs):
        grad = torch.autograd.grad(inputs=inputs, outputs=outputs,
                                   grad_outputs=torch.ones_like(outputs),
                                   create_graph=True, retain_graph=True)[0]
        grad_ = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        penalty = torch.mean((1. - grad_) ** 2)
        return penalty


def parseToMidi(pianoroll, midi_out_path, name="My_Track", jupyter_display_midi=False):
    assert(midi_out_path != None, "Incorrect out path")
    if not os.path.exists(midi_out_path):
        os.makedirs(midi_out_path)
    pianoroll = pianoroll.cpu().numpy().copy()
    pianoroll = pianoroll.transpose([1,0,2,3,4])
    pianoroll = pianoroll.reshape(4,-1,84)
    zeros_pad_a = np.zeros((*pianoroll.shape[:-1], 24))
    zeros_pad_b = np.zeros((*pianoroll.shape[:-1], 20))
    pianoroll = np.concatenate([zeros_pad_a, pianoroll, zeros_pad_b], axis=2)
    pianoroll = np.where(pianoroll > 0, 1, 0)
    tracks = []
    drums_track = pypi.BinaryTrack(name="Drums", program=0, is_drum=True, pianoroll=pianoroll[0])
    piano_track = pypi.BinaryTrack(name="Piano", program=0, is_drum=False, pianoroll=pianoroll[1])
    bass_track = pypi.BinaryTrack(name="Bass", program=33, is_drum=False, pianoroll=pianoroll[2])
    ensemble_track = pypi.BinaryTrack(name="Ensemble", program=48, is_drum=False, pianoroll=pianoroll[3])

    tracks = [drums_track, piano_track, bass_track, ensemble_track]
    m = pypi.Multitrack(name=name, tracks=tracks, resolution=4)
    mid_file_path = os.path.join(midi_out_path, f"{m.name}.mid")
    pypi.write(mid_file_path, m)
    if jupyter_display_midi:
        try:
            song = converter.parse(os.path.join(midi_out_path, f"{m.name}.mid"))
            song.show()
            song.show('midi')
        except Exception as e:
            print(e)
            # print(f'Failed to display {os.path.join(midi_out_path, f"{m.name}.mid")}')


def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)

def plot_losses(losses):
    fig, axs = plt.subplots(3, figsize=(20, 30))
    axs[0].plot(losses['gloss'], 'orange', label='generator')
    axs[0].plot(losses['closs'], 'm', label='critic')
    axs[0].legend(loc='best')
    axs[0].set_title('Critic/Gen Loss', fontsize=20)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)

    axs[1].plot(losses['crloss'], 'b', label='critic real')
    axs[1].plot(losses['cfloss'], 'r', label='critic fake')
    axs[1].set_title('Critic real/fake Loss', fontsize=20)
    axs[1].legend(loc='best')
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Loss', fontsize=12)

    axs[2].plot(losses['cploss'], 'black', label='critic penalty')
    axs[2].set_title('Gradient Penalty', fontsize=20)
    axs[2].set_xlabel('Epoch', fontsize=12)
    axs[2].set_ylabel('Loss', fontsize=12)
    plt.show()

def generate_samples_from_gen(gen, out_path, num_samples=1, sample_name="My Track"):

    for i in range(num_samples):
        chords = torch.randn(1, Z_DIM)
        style = torch.randn(1, Z_DIM)
        melody = torch.randn(1, N_TRACKS, Z_DIM)
        groove = torch.randn(1, N_TRACKS, Z_DIM)

        sample = gen(chords, style, melody, groove).detach()
        if(num_samples == 1):
            parseToMidi(sample, midi_out_path=out_path, name=sample_name)
        else:
            parseToMidi(sample, midi_out_path=out_path, name=f'{sample_name}_{i}')

def generate_samples(gen_path : str, out_path, num_samples=1, sample_name="My Track"):
    generator = torch.load(gen_path)
    generate_samples_from_gen(generator, out_path, num_samples, sample_name)