import torch 

def greedy_decoding():
    pass


def sampling_decoding():
    pass

def mixture_of_log_sampling(y, log_scale_min=-7.0, clamp_log_scale=False):
    nr_mix = y.shape[1] // 3

    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    temp = logit_probs.data.new(logit_probs.shape).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    one_hot = to_one_hot(argmax, nr_mix)

    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)

    u = means.data.new(means.shape).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x
