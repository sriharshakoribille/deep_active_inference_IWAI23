import numpy as np
import pickle
import torch

np_precision = np.float32

def make_batch_dsprites_random(game, index, size, repeats):
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    S0_real = np.zeros((size, 6), dtype=np_precision)
    S1_real = np.zeros((size, 6), dtype=np_precision)
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        o0[i] = game.current_frame(index)
        S0_real[i] = game.current_s[index,1:]
        S0_real[i,5] = game.last_r[index]
        Ppi = np.random.rand(4).astype(np_precision)
        Ppi /= np.sum(Ppi)
        pi0 = np.random.choice(4, p=Ppi)
        game.pi_to_action(pi0, index, repeats=repeats)
        pi_one_hot[i, pi0] = 1.0
        o1[i] = game.current_frame(index)
        S1_real[i] = game.current_s[index,1:]
        S1_real[i,5] = game.last_r[index]
    return torch.Tensor(o0).permute(0,3,1,2),\
          torch.Tensor(o1).permute(0,3,1,2), \
            torch.Tensor(pi_one_hot), S0_real, S1_real

def make_batch_dsprites_random_reward_transitions(game, index, size, deepness=1, repeats=1):
    '''
    Make a batch of random datapoints which are designed to test whether the
    agent understands the concept of reward changes..
    '''
    o0 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    o1 = np.zeros((size, 64, 64, 1), dtype=np_precision)
    pi0 = np.zeros((size),dtype=np.int32) # just 'up'
    pi_one_hot = np.zeros((size,4), dtype=np_precision)
    for i in range(size):
        game.randomize_environment(index)
        game.current_s[index,5] = 31 # Object located right at the edge of crossing.
        o0[i] = game.current_frame(index)
        for t in range(deepness):
            game.pi_to_action(pi0[i], index, repeats=repeats)
        pi_one_hot[i,pi0[i]] = 1.0
        o1[i] = game.current_frame(index)
    return torch.Tensor(o0).permute(0,3,1,2),\
          torch.Tensor(o1).permute(0,3,1,2), \
            torch.Tensor(pi_one_hot)

def softmax_multi_with_log(x, single_values=4, eps=1e-20, temperature=10.0):
    """Compute softmax values for each sets of scores in x."""
    x= x.reshape(-1, single_values)
    x = x - torch.max(x,1)[0].reshape(-1,1) # Normalization
    e_x = torch.exp(x/temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1,1)
    logSM = x - torch.log(e_x.sum(axis=1).reshape(-1,1) + eps) # to avoid infs
    return SM, logSM

def make_batch_dsprites_active_inference(games, model, deepness=10, samples=5, calc_mean=False, repeats=1, device='cpu'):
    o0 = games.current_frame_all()
    o0_repeated = torch.Tensor(o0.repeat(4,0)) # The 0th dimension
    o0 = torch.Tensor(o0).permute(0,3,1,2).to(device)
    o0_repeated = o0_repeated.permute(0,3,1,2).to(device)

    pi_one_hot = torch.eye(4)
    pi_repeated = torch.tile(pi_one_hot,(games.games_no, 1)).to(device)

    sum_G, sum_terms, po2 = model.calculate_G_repeated(o0_repeated, pi_repeated, steps=deepness, samples=samples, calc_mean=calc_mean)
    terms1 = -sum_terms[0]
    terms12 = -sum_terms[0]+sum_terms[1]
    # Shape now is (games_no,4)
    #Ppi, log_Ppi = softmax_multi_with_log(-terms1.numpy(), 4) # For agent driven just by reward
    #Ppi, log_Ppi = softmax_multi_with_log(-terms12.numpy(), 4) # For agent driven by terms 1 and 2
    Ppi, log_Ppi = softmax_multi_with_log(-sum_G, 4) # Full active inference agent

    pi_choices = np.array([np.random.choice(4,p=Ppi[i].detach().numpy()) for i in range(games.games_no)])

    # One hot version..
    pi0 = np.zeros((games.games_no,4), dtype=np_precision)
    pi0[np.arange(games.games_no), pi_choices] = 1.0

    # Apply the actions!
    for i in range(games.games_no): games.pi_to_action(pi_choices[i], i, repeats=repeats)
    o1 = games.current_frame_all()

    return o0, torch.Tensor(o1).permute(0,3,1,2), torch.Tensor(pi0), log_Ppi

def compare_reward(o1, po1):
    ''' Using MSE. '''
    logpo1 = np.square(o1[:,0:3,0:64,:] - po1[:,0:3,0:64,:]).mean(axis=(0,1,2,3))
    return logpo1
