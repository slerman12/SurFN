import torch
import numpy as np
import autograd_hacks


def get_fittest(policy, log_probs, advantages, selection_rate=0.04, seed=None):
    """
    Get boolean vector of whether each policy parameter belongs to the fittest.

    Args:
        policy: the actor model
        log_probs: the policy's batch of log probabilities for selected actions
        advantages: advantages corresponding with each log prob
        selection_rate: fraction of parameters that should belong to fittest and get saved
        seed: random seed
    """

    if hasattr(policy, 'surfn_hooks'):
        for handle in policy.surfn_hooks:
            handle.remove()
        del policy.surfn_hooks

    log_probs.sum().backward(retain_graph=True)
    autograd_hacks.compute_grad1(policy)

    # todo abs val instead of relu and softmax * advantages for fitness
    grads = torch.cat([param.grad1.flatten(start_dim=1) for param in policy.parameters()
                       if hasattr(param, 'grad1')], dim=1)
    credit = torch.relu(grads)

    credit_sum = credit.sum(dim=0)
    advantages = advantages.view([credit.shape[0]] + [1] * (len(credit.shape) - 1))
    credit_advantages_sum = torch.sum(credit * advantages, dim=0)

    fitness = credit_advantages_sum / credit_sum
    # todo compatibility with negative rewards?
    fitness[torch.isnan(fitness)] = 0  # todo -inf ?

    num_selected = max(1, int(selection_rate * fitness.shape[0]))
    # _, selected = torch.topk(fitness, num_selected)
    logits = fitness - min(fitness) + 0.001

    # todo abs, sep back on ent after cl, momentum,pro temporary/dec, softmax, crit, diff selrs, dec-r, r-ns,
    logits = logits ** 2

    logits = torch.ones_like(logits)

    # credit_sum[torch.isnan(fitness)] = 0
    # logits = credit_sum - min(credit_sum) + 0.001

    probas = logits / torch.sum(logits)  # todo mult to dirc
    selected = np.random.choice(a=fitness.shape[0], size=num_selected, replace=False, p=probas.numpy())
    # seed is not None and torch.random.manual_seed (1010)
    # selected = logits.multinomial(num_samples=num_selected, replacement=False)  # changes seed
    # seed is not None and torch.random.manual_seed(seed)
    is_fittest = torch.zeros(fitness.shape[0], dtype=torch.bool)
    is_fittest[selected] = True

    handles = []
    fitness_size = 0
    for param in policy.parameters():
        if hasattr(param, 'grad1'):
            param_fittest = is_fittest[fitness_size:fitness_size + np.prod(param.shape)].view(param.shape)
            handle = param.register_hook(get_hook(param_fittest))
            handles.append(handle)
            fitness_size += np.prod(param.shape)

    policy.__dict__.setdefault('surfn_hooks', []).extend(handles)


def get_hook(is_fittest):
    """
    Get hook to nullify/freeze gradients for the fittest policy parameters.

    Args:
        is_fittest: boolean vector of whether each policy parameter belongs to the fittest
    """

    def hook(grad):
        grad = grad.clone()
        grad[is_fittest] = 0
        return grad
    return hook
