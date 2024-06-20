import torch
import numpy as np

from src_torch.torchutils import *

def compute_omega(kl_pi, a, b, c, d):
    return a * ( 1.0 - 1.0/(1.0 + torch.exp(- (kl_pi-b) / c)) ) + d

def compute_loss_top(model_top, s, log_Ppi, device='cpu'):
    _, Qpi, log_Qpi = model_top(s.to(device))

    # TERM: Eqs D_kl[Q(pi|s0)||P(pi)], where P(pi) = U(0,5) - Categorical K-L divergence
    # ----------------------------------------------------------------------
    kl_div_pi_anal = Qpi*(log_Qpi - log_Ppi)
    kl_div_pi = torch.sum(kl_div_pi_anal, 1)

    F_top = kl_div_pi
    return F_top, kl_div_pi, kl_div_pi_anal, Qpi

def compute_loss_mid(model_mid, s0, Ppi_sampled, qs1_mean, qs1_logvar, omega, device='cpu'):
    ps1, ps1_mean, ps1_logvar = model_mid.transition_with_sample(Ppi_sampled.to(device), s0.to(device))

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = torch.sum(kl_div_s_anal, 1)

    F_mid = kl_div_s
    loss_terms = (kl_div_s, kl_div_s_anal)
    return F_mid, loss_terms, ps1, ps1_mean, ps1_logvar

def compute_loss_down(model_down, o1, ps1_mean, ps1_logvar, omega, displacement = 0.00001, device='cpu'):
    qs1_mean, qs1_logvar = model_down.encoder(o1.to(device))
    qs1 = model_down.reparameterize(qs1_mean, qs1_logvar)
    po1 = model_down.decoder(qs1.to(device))

    # TERM: Eq[log P(o1|s1)]
    # --------------------------------------------------------------------------
    bin_cross_entr = o1 * torch.log(displacement + po1) + (1 - o1) * torch.log(displacement + 1 - po1) # Binary Cross Entropy
    logpo1_s1 = torch.sum(bin_cross_entr, dim=[1,2,3])

    # TERM: Eqpi D_kl[Q(s1)||N(0.0,1.0)]
    # --------------------------------------------------------------------------
    kl_div_s_naive_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, torch.Tensor([0.0]), torch.Tensor([0.0]), omega)
    kl_div_s_naive = torch.sum(kl_div_s_naive_anal, 1)

    # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
    # ----------------------------------------------------------------------
    kl_div_s_anal = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega)
    kl_div_s = torch.sum(kl_div_s_anal, 1)

    if model_down.gamma <= 0.05:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*kl_div_s_naive
    elif model_down.gamma >= 0.95:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*kl_div_s
    else:
       F = - model_down.beta_o*logpo1_s1 + model_down.beta_s*(model_down.gamma*kl_div_s + (1.0-model_down.gamma)*kl_div_s_naive)
    loss_terms = (-logpo1_s1, kl_div_s, kl_div_s_anal, kl_div_s_naive, kl_div_s_naive_anal)
    return F, loss_terms, po1, qs1

def train_model_top(model_top, s, log_Ppi, optimizer, device='cpu'):
    s_stopped = s.detach()
    log_Ppi_stopped = log_Ppi.detach()
    optimizer.zero_grad()
    F, kl_pi, _, _ = compute_loss_top(model_top=model_top, s=s_stopped, log_Ppi=log_Ppi_stopped, device=device)
    F.sum().backward()
    optimizer.step()
    return kl_pi.cpu()

def train_model_mid(model_mid, s0, qs1_mean, qs1_logvar, Ppi_sampled, omega, optimizer, device='cpu'):
    s0_stopped = s0.detach()
    qs1_mean_stopped = qs1_mean.detach()
    qs1_logvar_stopped = qs1_logvar.detach()
    Ppi_sampled_stopped = Ppi_sampled.detach()
    omega_stopped = omega.detach()
    optimizer.zero_grad()
    F, loss_terms, ps1, ps1_mean, ps1_logvar = compute_loss_mid(model_mid=model_mid, s0=s0_stopped, Ppi_sampled=Ppi_sampled_stopped, qs1_mean=qs1_mean_stopped, qs1_logvar=qs1_logvar_stopped, omega=omega_stopped, device=device)
    F.sum().backward()
    optimizer.step()
    return ps1_mean.cpu(), ps1_logvar.cpu()

def train_model_down(model_down, o1, ps1_mean, ps1_logvar, omega, optimizer, device='cpu'):
    ps1_mean_stopped = ps1_mean.detach()
    ps1_logvar_stopped = ps1_logvar.detach()
    omega_stopped = omega.detach()
    optimizer.zero_grad()
    F, _, _, _ = compute_loss_down(model_down=model_down, o1=o1, ps1_mean=ps1_mean_stopped, ps1_logvar=ps1_logvar_stopped, omega=omega_stopped, device=device)
    F.sum().backward()
    optimizer.step()