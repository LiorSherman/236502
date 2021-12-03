import torch
import os
import argparse
from torch import nn
from gan.settings import *
import pandas as pd
from gan.dataset import Dataset
from torch.utils.data import DataLoader
from gan.gan import MidiGenerator, MidiCritic
from gan.utils import initialize_weights, generate_samples_from_gen
from gan.train import Trainer


def desc():
    return 'Training Gan'


def execute():
    parser = argparse.ArgumentParser(description='GAN : Training Gan', prog=__name__)
    parser.add_argument('dataset', help='path to the npy data set file')
    parser.add_argument('model_name', default='My Trained Model', help='model name')
    parser.add_argument('--limit_dataset', type=int, default=0, help='limit dataset number of samples')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number of epochs')
    parser.add_argument('--sample_step', type=int, default=0, help='generate samples every X epoch')
    parser.add_argument('--sample_num', type=int, default=5, help='number of samples to generate every time')
    parser.add_argument('--generate_final_num', type=int, default=1, help='number of samples to generate at the end '
                                                                          'of training')
    parser.add_argument('--opt', default='Adam', help='experience with different optimizers')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')


    args = parser.parse_args()

    # check for args validity
    if not os.path.exists(args.dataset):
        raise ValueError(f"{args.dataset} is not a valid dataset dir path")

    model_path = os.path.join('./gan/my_trained_models', args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.limit_dataset:
        dataset = Dataset(file=args.dataset, max_num_files=args.limit_dataset)
    else:
        dataset = Dataset(file=args.dataset)

    if (args.batch_size > len(dataset)):
        raise ValueError(f"Dataset size file is too small. Requires at least {args.batch_size} sizes")

    print('Loading Dataset...')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('Loading Models...')
    generator = MidiGenerator(z_dim=32, hid_channels=1024, hid_features=1024, out_channels=1).to(device)

    # g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.9))
    critic = MidiCritic(hid_channels=128,
                        hid_features=1024,
                        out_features=1).to(device)
    # c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001, betas=(0.5, 0.9))

    print(f'Chosen optimizer : {args.opt}')

    if args.opt == 'RMSprop':
        g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
        c_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
    elif args.opt == 'SGD':
        g_optimizer = torch.optim.SGD(generator.parameters(), lr=args.lr)
        c_optimizer = torch.optim.SGD(critic.parameters(), lr=args.lr)
    else: #'Adam' by default
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
        c_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.9))

    generator = generator.apply(initialize_weights)
    critic = critic.apply(initialize_weights)
    print('Setting Up Training...')
    trainer = Trainer(generator, critic, g_optimizer, c_optimizer, model_path, device)
    print('Begin Training...')
    if args.sample_step:
        trainer.train(dataloader, batch_size=args.batch_size, epochs=args.epochs, device=device,
                      sample_step=args.sample_step, sample_num=args.sample_num)
    else:
        trainer.train(dataloader, batch_size=args.batch_size, epochs=args.epochs, device=device)
    print('Training finished successfully')

    print('Saving Model')
    generator = generator.eval().cpu()
    critic = critic.eval().cpu()

    torch.save(generator.state_dict(), os.path.join(model_path, f'generator_e{args.epochs}_s{len(dataset)}.pt'))
    torch.save(critic.state_dict(), os.path.join(model_path, f'critic_e{args.epochs}_s{len(dataset)}.pt'))
    losses = trainer.data.copy()
    df = pd.DataFrame.from_dict(losses)
    df.to_csv(os.path.join(model_path, 'results.csv'), index=False)
    print('Generating samples...')
    generate_samples_from_gen(generator, model_path, args.generate_final_num)
    print('Finished')


