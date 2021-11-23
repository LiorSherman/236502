import torch
from torch import nn
from tqdm.notebook import tqdm
from gan.utils import WassersteinLoss, GradientPenalty, parseToMidi
from gan.settings import N_TRACKS
import os

#_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer():
    """
    Wrapper Training class
    """
    def __init__(self, generator, critic, g_optimizer, c_optimizer, out_path, device):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.gen_opt = g_optimizer
        self.critic_opt = c_optimizer
        self.gen_criterion = WassersteinLoss().to(device)
        self.critic_criterion = WassersteinLoss().to(device)
        self.critic_penalty = GradientPenalty().to(device)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_path = out_path

    def _sample_during_training_samples(self, epoch, device, sample_step=0, sample_num=5):
        """
        generates samples during training
        :param epoch: number of current epoch
        :param device: training device (cpu/cuda)
        :param sample_step: generate samples every 'sample_step'. If set to 0 - no samples during training
        :param sample_num: number of samples to generate every time
        :return:
        """

        if not sample_step or epoch % sample_step != 0:
            return

        training_samples_out_path = os.path.join(self.out_path, 'training samples')
        if not os.path.exists(training_samples_out_path):
            os.makedirs(training_samples_out_path)
        for i in range(0, sample_num):
            cords = torch.randn(1, 32).to(device)
            style = torch.randn(1, 32).to(device)
            melody = torch.randn(1, N_TRACKS, 32).to(device)
            groove = torch.randn(1, N_TRACKS, 32).to(device)
            with torch.no_grad():
                self.generator.eval()
                sample = self.generator(cords, style, melody, groove).detach()
                parseToMidi(sample, midi_out_path=training_samples_out_path, name=f'sample_{epoch}_{i}')
                self.generator.train()
        pass

    def train(self, dataloader, epochs=500, batch_size=64, repeat=5, display_step=10, device='cpu', **kwargs):
        """
        Trains model on a given dataset
        :param dataloader: dataloader containing proper dataset
        :param epochs: total numbers of epochs to train
        :param batch_size: num of samples per batch
        :param repeat: number of steps to repeat descriminator (critic) train over the generator each epoch
        :param display_step: display loss message values every 'display_step'
        :param device: device to train on (cpu/cuda)
        :param kwargs: params for sampling and exporting generated samples during training. see _sample_during_training_samples
        :return: void
        """
        print(f'Begining Train on device: {device}')
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(device)
        self.data = {'gloss': [], 'closs': [], 'cfloss': [], 'crloss': [], 'cploss': []}
        for epoch in tqdm(range(epochs)):
            exp_gen_loss = 0
            expected_critic_loss = 0
            exp_critic_fake_loss = 0
            exp_critic_real_loss = 0
            exp_penalty_loss = 0
            for real in dataloader:
                real = real.to(device)
                # Train Critic
                batch_critic_loss = 0
                batch_critic_fake_loss = 0
                batch_critic_real_loss = 0
                batch_penalty_loss = 0
                for _ in range(repeat):

                    # create random noise
                    cords = torch.randn(batch_size, 32).to(device)
                    style = torch.randn(batch_size, 32).to(device)
                    melody = torch.randn(batch_size, N_TRACKS, 32).to(device)
                    groove = torch.randn(batch_size, N_TRACKS, 32).to(device)

                    self.critic_opt.zero_grad()

                    # generate fake samples (critic train, no grad for generator)
                    with torch.no_grad():
                        fake = self.generator(cords, style, melody, groove).detach()

                    # critic forward + loss calculation on fake samples
                    fake_pred = self.critic(fake)
                    fake_loss = self.critic_criterion(fake_pred, -torch.ones_like(fake_pred))

                    # critic forward + loss calculation on real samples
                    real_pred = self.critic(real)
                    real_loss = self.critic_criterion(real_pred, torch.ones_like(real_pred))

                    # gradient penalty calculation
                    realfake = self.alpha * real + (1. - self.alpha) * fake
                    realfake_pred = self.critic(realfake)
                    penalty = self.critic_penalty(realfake, realfake_pred)

                    # critic loss
                    critic_w_loss = fake_loss + real_loss
                    closs = critic_w_loss + 10 * penalty

                    closs.backward(retain_graph=True)
                    self.critic_opt.step()

                    batch_critic_fake_loss += fake_loss.item() / repeat
                    batch_critic_real_loss += real_loss.item() / repeat
                    batch_penalty_loss += 10 * penalty.item() / repeat
                    batch_critic_loss += critic_w_loss.item() / repeat

                exp_critic_fake_loss += batch_critic_fake_loss / len(dataloader)
                exp_critic_real_loss += batch_critic_real_loss / len(dataloader)
                exp_penalty_loss += batch_penalty_loss / len(dataloader)
                expected_critic_loss += batch_critic_loss / len(dataloader)

                # Train Generator
                # create random noise
                self.gen_opt.zero_grad()
                cords = torch.randn(batch_size, 32).to(device)
                style = torch.randn(batch_size, 32).to(device)
                melody = torch.randn(batch_size, N_TRACKS, 32).to(device)
                groove = torch.randn(batch_size, N_TRACKS, 32).to(device)

                # generate samples
                fake = self.generator(cords, style, melody, groove)

                # critic forward pass + loss calculation for generator
                fake_pred = self.critic(fake)
                b_gloss = self.gen_criterion(fake_pred, torch.ones_like(fake_pred))
                b_gloss.backward()
                self.gen_opt.step()
                exp_gen_loss += b_gloss.item() / len(dataloader)

            self.data['gloss'].append(exp_gen_loss)
            self.data['closs'].append(expected_critic_loss)
            self.data['cfloss'].append(exp_critic_fake_loss)
            self.data['crloss'].append(exp_critic_real_loss)
            self.data['cploss'].append(exp_penalty_loss)

            if epoch % display_step == 0:
                print(f'Epoch {epoch}/{epochs} | Generator loss: {exp_gen_loss:.3f} | ' \
                      + f'Critic loss: {expected_critic_loss:.3f} (fake: {exp_critic_fake_loss:.3f}, real: {exp_critic_real_loss:.3f}, penalty: {exp_penalty_loss:.3f})')


            self._sample_during_training_samples(epoch, device, **kwargs)
                # for i in range(0, sample_num):
                #     cords = torch.randn(1, 32).to(device)
                #     style = torch.randn(1, 32).to(device)
                #     melody = torch.randn(1, N_TRACKS, 32).to(device)
                #     groove = torch.randn(1, N_TRACKS, 32).to(device)
                #     with torch.no_grad():
                #         self.generator.eval()
                #         sample = self.generator(cords, style, melody, groove).detach()
                #         parseToMidi(sample, midi_out_path=save_out_dir, name=f'sample_{epoch}_{i}')
                #         self.generator.train()


