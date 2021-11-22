import torch
from torch import nn
from gan.settings import N_BARS, N_STEPS_PER_BAR, N_TRACKS


class Reshape(nn.Module):
    def __init__(self, shape=[32, 1, 1]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, *self.shape)


class TempGenerator(nn.Module):
    def __init__(self,
                 z_dim: int = 32,
                 hid_channels: int = 1024):
        super().__init__()
        self.n_bars = N_BARS
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # input shape: (batch_size, z_dim)
            Reshape(shape=[z_dim, 1, 1]),
            # output shape: (batch_size, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, hid_channels,
                               kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2, 1)
            nn.ConvTranspose2d(hid_channels, z_dim,
                               kernel_size=(self.n_bars - 1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, z_dimension, 1, 1)
            Reshape(shape=[z_dim, self.n_bars])
        )

    def forward(self, x):
        y = self.net(x)
        return y


class BarGenerator(nn.Module):
    def __init__(self,
                 z_dim: int = 32,
                 hid_features: int = 1024,
                 hid_channels: int = 512,
                 out_channels: int = 1):
        super().__init__()
        self.n_steps_per_bar = N_STEPS_PER_BAR
        self.n_pitches = 84
        self.net = nn.Sequential(
            # input shape: (batch_size, 4*z_dimension)
            nn.Linear(4 * z_dim, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_features) #64x1024
            Reshape(shape=[hid_channels, hid_features // hid_channels, 1]),
            # output shape: (batch_size, hid_channels, hid_features//hid_channels, 1) #64x512x2x1

            nn.ConvTranspose2d(hid_channels, hid_channels,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2*hid_features//hid_channels, 1) 64x512x4x1

            nn.ConvTranspose2d(hid_channels, hid_channels // 2,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 4*hid_features//hid_channels, 1)   64x256x8x1

            nn.ConvTranspose2d(hid_channels // 2, hid_channels // 2,
                               kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 1)  64x256x16x1

            nn.ConvTranspose2d(hid_channels // 2, hid_channels // 2,
                               kernel_size=(1, 7), stride=(1, 7), padding=0),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 7) 64x256x16x7

            nn.ConvTranspose2d(hid_channels // 2, out_channels,
                               kernel_size=(1, 12), stride=(1, 12), padding=0),
            # output shape: (batch_size, out_channels, 8*hid_features//hid_channels, n_pitches) 64x1x16x84
            Reshape(shape=[1, 1, self.n_steps_per_bar, self.n_pitches])
            # output shape: (batch_size, out_channels, 1, n_steps_per_bar, n_pitches)
        )

    def forward(self, x):
        y = self.net(x)
        return y


class MidiGenerator(nn.Module):

    def __init__(self,
                 z_dim=32,
                 hid_channels=1024,
                 hid_features=1024,
                 out_channels=1):
        super().__init__()
        self.n_tracks = N_TRACKS
        self.n_bars = N_BARS
        self.n_steps_per_bar = N_STEPS_PER_BAR
        self.n_pitches = 84

        # chords generator
        self.chords_net = TempGenerator(z_dim=z_dim, hid_channels=hid_channels)

        # melody generators
        self.melody_net = nn.ModuleDict(
            {f'melody_gen_{n}':
                 TempGenerator(z_dim=z_dim, hid_channels=hid_channels)
             for n in range(self.n_tracks)}
        )

        # bar generators
        self.bar_net = nn.ModuleDict(
            {f'bar_gen_{n}':
                 BarGenerator(z_dim=z_dim, hid_features=hid_features, hid_channels=hid_channels // 2,
                              out_channels=out_channels)
             for n in range(self.n_tracks)}
        )

    def forward(self, chords, style, melody, groove):
        # input shapes
        # chords shape: (batch_size, z_dimension)
        # style shape: (batch_size, z_dimension)
        # melody shape: (batch_size, n_tracks, z_dimension)
        # groove shape: (batch_size, n_tracks, z_dimension)

        # out shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)

        # get style_out - Track Independent, Bar Independent
        style_out = style

        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []

            # get chord_out - Track Independent, Bar Dependent
            chord_bar_out = self.chords_net(chords)[:, :, bar]

            for track in range(self.n_tracks):
                # get melody_out - Track Dependent, Bar Dependent
                melody_track_in = melody[:, track, :]
                melody_gen = self.melody_net[f'melody_gen_{track}']
                melody_track_out = melody_gen(melody_track_in)
                melody_track_bar_out = melody_track_out[:, :, bar]

                # get groove_out - Track Dependent, Bar Independent
                groove_tack_out = groove[:, track, :]

                # concat outputs
                bar_gen_in = torch.cat([chord_bar_out, style_out, melody_track_bar_out, groove_tack_out], dim=1)

                bar_gen = self.bar_net[f'bar_gen_{track}']
                bar_gen_out = bar_gen(bar_gen_in)
                track_outs.append(bar_gen_out)

            track_out = torch.cat(track_outs, dim=1)

            # appends a bar for each track
            bar_outs.append(track_out)

        out = torch.cat(bar_outs, dim=2)
        # out shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
        return out

    def generate(self, size, requires_grad=False, device='cuda:0'):
        cords = torch.randn(size, 32).to(device)
        style = torch.randn(size, 32).to(device)
        melody = torch.randn(size, self.n_tracks, 32).to(device)
        groove = torch.randn(size, self.n_tracks, 32).to(device)
        with torch.no_grad:
            out = self.forward(cords, style, melody, groove)

        return out


class MidiCritic(nn.Module):
    def __init__(self,
                 hid_channels: int = 128,
                 hid_features: int = 1024,
                 out_features: int = 1):
        super().__init__()
        self.n_tracks = N_TRACKS
        self.n_bars = N_BARS
        self.n_steps_per_bar = N_STEPS_PER_BAR
        self.n_pitches = 84
        self.net = nn.Sequential(
            # input shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
            nn.Conv3d(self.n_tracks, hid_channels, (2, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches)
            nn.Conv3d(hid_channels, hid_channels, (self.n_bars - 1, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches)
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 12), (1, 1, 12), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 7), (1, 1, 7), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//2, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//4, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//4, n_pitches//12)
            nn.Conv3d(hid_channels, 2 * hid_channels, (1, 4, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//8, n_pitches//12)
            nn.Conv3d(2 * hid_channels, 4 * hid_channels, (1, 3, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//16, n_pitches//12)
            nn.Flatten(),
            nn.Linear(4 * hid_channels, hid_features),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_features)
            nn.Linear(hid_features, out_features),
            # output shape: (batch_size, out_features)
        )

    def forward(self, x):
        y = self.net(x)
        return y
