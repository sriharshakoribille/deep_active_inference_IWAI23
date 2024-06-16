import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from shutil import copyfile
from src_torch.torchutils import *

class ModelTop(nn.Module):
    def __init__(self, s_dim, pi_dim, precision):
        super(ModelTop, self).__init__()

        self.precision = precision
        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.qpi_net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pi_dim)
        )

    def encode_s(self, s0):
        logits_pi = self.qpi_net(s0)
        q_pi = F.softmax(logits_pi, dim=1)
        log_q_pi = torch.log(q_pi + 1e-20)
        return logits_pi, q_pi, log_q_pi

class ModelMid(nn.Module):
    def __init__(self, s_dim, pi_dim, precision):
        super(ModelMid, self).__init__()

        self.precision = precision
        self.s_dim = s_dim
        self.pi_dim = pi_dim

        self.ps_net = nn.Sequential(
            nn.Linear(pi_dim + s_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(s_dim + s_dim, s_dim + s_dim)
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def transition(self, pi, s0):
        mean, logvar = torch.split(self.ps_net(torch.cat([pi, s0], 1)), split_size_or_sections=2, dim=1)
        return mean, logvar

    def transition_with_sample(self, pi, s0):
        ps1_mean, ps1_logvar = self.transition(pi, s0)
        ps1 = self.reparameterize(ps1_mean, ps1_logvar)
        return ps1, ps1_mean, ps1_logvar

class ModelDown(nn.Module):
    def __init__(self, s_dim, pi_dim, precision, colour_channels, resolution):
        super(ModelDown, self).__init__()

        self.precision = precision
        self.s_dim = s_dim
        self.pi_dim = pi_dim
        self.colour_channels = colour_channels
        self.resolution = resolution
        if self.resolution == 64:
            last_strides = 2
        elif self.resolution == 32:
            last_strides = 1
        else:
            exit('Unknown resolution..')

        self.qs_net = nn.Sequential(
            nn.Conv2d(colour_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(s_dim + s_dim, s_dim + s_dim)
        )

        self.po_net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=last_strides, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, colour_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def encoder(self, o):
        mean_s, logvar_s = torch.split(self.qs_net(o), split_size_or_sections=2, dim=1)
        return mean_s, logvar_s

    def decoder(self, s):
        po = self.po_net(s)
        return po

    def encoder_with_sample(self, o):
        mean, logvar = self.encoder(o)
        s = self.reparameterize(mean, logvar)
        return s, mean, logvar


class ActiveInferenceModel:
    def __init__(self, s_dim, pi_dim, gamma, beta_s, beta_o, colour_channels=1, resolution=64):

        self.precision = 'float32'
        self.s_dim = s_dim
        self.pi_dim = pi_dim

        if self.pi_dim > 0:
            self.model_top = ModelTop(s_dim, pi_dim, self.precision)
            self.model_mid = ModelMid(s_dim, pi_dim, self.precision)
        self.model_down = ModelDown(s_dim, pi_dim, self.precision, colour_channels, resolution)

        self.model_down.beta_s = nn.Parameter(torch.tensor(beta_s, dtype=torch.float32), requires_grad=False)
        self.model_down.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=False)
        self.model_down.beta_o = nn.Parameter(torch.tensor(beta_o, dtype=torch.float32), requires_grad=False)
        self.pi_one_hot = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
                                       requires_grad=False)
        self.pi_one_hot_3 = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 1.0]], dtype=torch.float32),
                                         requires_grad=False)

    def save_weights(self, folder_chp):
        torch.save(self.model_down.qs_net.state_dict(), folder_chp + '/checkpoint_qs')
        torch.save(self.model_down.po_net.state_dict(), folder_chp + '/checkpoint_po')
        if self.pi_dim > 0:
            torch.save(self.model_top.qpi_net.state_dict(), folder_chp + '/checkpoint_qpi')
            torch.save(self.model_mid.ps_net.state_dict(), folder_chp + '/checkpoint_ps')

    def load_weights(self, folder_chp):
        self.model_down.qs_net.load_state_dict(torch.load(folder_chp + '/checkpoint_qs'))
        self.model_down.po_net.load_state_dict(torch.load(folder_chp + '/checkpoint_po'))
        if self.pi_dim > 0:
            self.model_top.qpi_net.load_state_dict(torch.load(folder_chp + '/checkpoint_qpi'))
            self.model_mid.ps_net.load_state_dict(torch.load(folder_chp + '/checkpoint_ps'))

    def save_all(self, folder_chp, stats, script_file="", optimizers={}):
        self.save_weights(folder_chp)
        with open(folder_chp + '/stats.pkl', 'wb') as ff:
            pickle.dump(stats, ff)
        checkpoint = {
            'optim': optimizers
        }
        torch.save(checkpoint, folder_chp + '/checkpoint')
        copyfile('src/tfmodel.py', folder_chp + '/tfmodel.py')
        copyfile('src/tfloss.py', folder_chp + '/tfloss.py')

    def load_all(self, folder_chp):
        self.load_weights(folder_chp)
        with open(folder_chp + '/stats.pkl', 'rb') as ff:
            stats = pickle.load(ff)
        try:
            checkpoint = torch.load(folder_chp + '/checkpoint')
            optimizers = checkpoint['optim']
        except:
            optimizers = {}
        if len(stats['var_beta_s']) > 0:
            self.model_down.beta_s.data = torch.tensor(stats['var_beta_s'][-1], dtype=torch.float32)
        if len(stats['var_gamma']) > 0:
            self.model_down.gamma.data = torch.tensor(stats['var_gamma'][-1], dtype=torch.float32)
        if len(stats['var_beta_o']) > 0:
            self.model_down.beta_o.data = torch.tensor(stats['var_beta_o'][-1], dtype=torch.float32)
        return stats, optimizers

    def check_reward(self, o):
        if self.model_down.resolution == 64:
            return torch.mean(calc_reward(o), dim=[1, 2, 3]) * 10.0
        elif self.model_down.resolution == 32:
            # return torch.sum(calc_reward_animalai(o), dim=[1, 2, 3])
            print('Not implemented yet..')

    def imagine_future_from_o(self, o0, pi):
        s0, _, _ = self.model_down.encoder_with_sample(o0)
        ps1, _, _ = self.model_mid.transition_with_sample(pi, s0)
        po1 = self.model_down.decoder(ps1)
        return po1

    def habitual_net(self, o):
        qs0_mean, _ = self.model_down.encoder(o)
        _, Qpi, _ = self.model_top.encode_s(qs0_mean)
        return Qpi

    def calculate_G_repeated(self, o, pi, steps=1, calc_mean=False, samples=10):
        s0 = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1 = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1_mean = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1_logvar = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        pi0 = torch.zeros((o.shape[0], steps, self.pi_dim), dtype=torch.float32)

        s0[:, 0] = self.model_down.encoder_with_sample(o)[0]
        Qpi_t_to_return = self.model_top.encode_s(s0[:, 0].reshape(-1, self.s_dim))[1].detach().numpy()

        pi0[:, 0, np.random.choice(self.pi_dim, p=Qpi_t_to_return)] = 1.0
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[:, 0].reshape(-1, self.pi_dim), s0[:, 0].reshape(-1, self.s_dim))
        ps1[:, 0] = ps1_new.detach().numpy()
        ps1_mean[:, 0] = ps1_mean_new.detach().numpy()
        ps1_logvar[:, 0] = ps1_logvar_new.detach().numpy()

        if 1 < steps:
            if calc_mean:
                s0[:, 1] = ps1_mean_new.detach().numpy()
            else:
                s0[:, 1] = ps1_new.detach().numpy()

        for t in range(1, steps):
            pi0[:, t, np.random.choice(self.pi_dim, p=self.model_top.encode_s(s0[:, t].reshape(-1, self.s_dim))[1].detach().numpy())] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[:, t].reshape(-1, self.pi_dim), s0[:, t].reshape(-1, self.s_dim))
            ps1[:, t] = ps1_new.detach().numpy()
            ps1_mean[:, t] = ps1_mean_new.detach().numpy()
            ps1_logvar[:, t] = ps1_logvar_new.detach().numpy()

            if t + 1 < steps:
                if calc_mean:
                    s0[:, t + 1] = ps1_mean_new.detach().numpy()
                else:
                    s0[:, t + 1] = ps1_new.detach().numpy()

        G = torch.mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).detach().numpy()
        return G, pi0, Qpi_t_to_return

    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        s0 = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1 = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1_mean = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        ps1_logvar = torch.zeros((o.shape[0], steps, self.s_dim), dtype=torch.float32)
        pi0 = torch.zeros((o.shape[0], steps, self.pi_dim), dtype=torch.float32)

        s0[:, 0] = self.model_down.encoder_with_sample(o)[0]
        Qpi_t_to_return = self.model_top.encode_s(s0[:, 0].reshape(-1, self.s_dim))[1].detach().numpy()

        pi0[:, 0, np.random.choice(self.pi_dim, p=Qpi_t_to_return)] = 1.0
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[:, 0].reshape(-1, self.pi_dim), s0[:, 0].reshape(-1, self.s_dim))
        ps1[:, 0] = ps1_new.detach().numpy()
        ps1_mean[:, 0] = ps1_mean_new.detach().numpy()
        ps1_logvar[:, 0] = ps1_logvar_new.detach().numpy()

        if 1 < steps:
            if calc_mean:
                s0[:, 1] = ps1_mean_new.detach().numpy()
            else:
                s0[:, 1] = ps1_new.detach().numpy()

        for t in range(1, steps):
            pi0[:, t, np.random.choice(self.pi_dim, p=self.model_top.encode_s(s0[:, t].reshape(-1, self.s_dim))[1].detach().numpy())] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[:, t].reshape(-1, self.pi_dim), s0[:, t].reshape(-1, self.s_dim))
            ps1[:, t] = ps1_new.detach().numpy()
            ps1_mean[:, t] = ps1_mean_new.detach().numpy()
            ps1_logvar[:, t] = ps1_logvar_new.detach().numpy()

            if t + 1 < steps:
                if calc_mean:
                    s0[:, t + 1] = ps1_mean_new.detach().numpy()
                else:
                    s0[:, t + 1] = ps1_new.detach().numpy()

        G = torch.mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).detach().numpy()
        return G, pi0

    def calculate_G(self, s0, pi0, samples=10):
        term0 = torch.zeros(s0.shape[0], dtype=torch.float32)
        term1 = torch.zeros(s0.shape[0], dtype=torch.float32)

        for _ in range(samples):
            ps1, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
            po1 = self.model_down.decoder(ps1)
            qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

            # E [ log P(o|pi) ]
            logpo1 = self.check_reward(po1)
            term0 += logpo1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]
            term1 += -torch.sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), dim=1)

        term0 /= float(samples)
        term1 /= float(samples)

        term2_1 = torch.zeros(s0.shape[0], dtype=torch.float32)
        term2_2 = torch.zeros(s0.shape[0], dtype=torch.float32)

        for _ in range(samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[0])
            term2_1 += torch.sum(entropy_bernoulli(po1_temp1), dim=[1, 2, 3])

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2_2 += torch.sum(entropy_bernoulli(po1_temp2), dim=[1, 2, 3])

        term2_1 /= float(samples)
        term2_2 /= float(samples)

        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2

        return G, [term0, term1, term2], ps1, ps1_mean, po1

    def calculate_G_mean(self, s0, pi0):
        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1 = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        logpo1 = self.check_reward(po1)
        term0 = logpo1

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - torch.sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), dim=1)

        # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
        term2_1 = torch.sum(entropy_bernoulli(po1_temp1), dim=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = torch.sum(entropy_bernoulli(po1_temp2), dim=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = - term0 + term1 + term2

        return G, [term0, term1, term2], ps1_mean, po1

    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        po1 = self.model_down.decoder(ps1_traj)
        qs1, _, qs1_logvar = self.model_down.encoder_with_sample(po1)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(po1)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = - torch.sum(entropy_normal_from_logvar(ps1_logvar_traj) + entropy_normal_from_logvar(qs1_logvar), dim=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.model_down.decoder(self.model_mid.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = torch.sum(entropy_bernoulli(po1_temp1), dim=[1,2,3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.model_down.decoder(self.model_down.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = torch.sum(entropy_bernoulli(po1_temp2), dim=[1,2,3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return - term0 + term1 + term2

    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = torch.zeros((depth, self.s_dim), dtype=torch.float32)
        ps1 = torch.zeros((depth, self.s_dim), dtype=torch.float32)
        ps1_mean = torch.zeros((depth, self.s_dim), dtype=torch.float32)
        ps1_logvar = torch.zeros((depth, self.s_dim), dtype=torch.float32)
        pi0 = torch.zeros((depth, self.pi_dim), dtype=torch.float32)

        s0[0] = starting_s
        try:
            Qpi_t_to_return = self.model_top.encode_s(s0[0].reshape(1,-1))[1].detach().numpy()[0]
            pi0[0, np.random.choice(self.pi_dim, p=Qpi_t_to_return)] = 1.0
        except:
            pi0[0, 0] = 1.0
            Qpi_t_to_return = pi0[0]
        ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[0].reshape(1,-1), s0[0].reshape(1,-1))
        ps1[0] = ps1_new[0].detach().numpy()
        ps1_mean[0] = ps1_mean_new[0].detach().numpy()
        ps1_logvar[0] = ps1_logvar_new[0].detach().numpy()
        if 1 < depth:
            if use_means:
                s0[1] = ps1_mean_new[0].detach().numpy()
            else:
                s0[1] = ps1_new[0].detach().numpy()
        for t in range(1, depth):
            try:
                pi0[t, np.random.choice(self.pi_dim, p=self.model_top.encode_s(s0[t].reshape(1,-1))[1].detach().numpy()[0])] = 1.0
            except:
                pi0[t, 0] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.model_mid.transition_with_sample(pi0[t].reshape(1,-1), s0[t].reshape(1,-1))
            ps1[t] = ps1_new[0].detach().numpy()
            ps1_mean[t] = ps1_mean_new[0].detach().numpy()
            ps1_logvar[t] = ps1_logvar_new[0].detach().numpy()
            if t+1 < depth:
                if use_means:
                    s0[t+1] = ps1_mean_new[0].detach().numpy()
                else:
                    s0[t+1] = ps1_new[0].detach().numpy()

        G = torch.mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).detach().numpy()
        return G, pi0, Qpi_t_to_return
