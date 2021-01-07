import torch
import numpy as np
import autograd_hacks


def set_fittest(policy, log_probs, advantages, optimizer, probas=None, selection_rate=0.04, min_adv=0,
                gradient_agg=False, do_surfn=True, uniform_sampling=False, deterministic=False,
                credit_method="bt-gc", credit_method_2="relu", fitness_dist="None", probas_dist="sq", temp=0.2,
                nonzero=True, select_method="numpy"):
    """
    Get boolean vector of whether each policy parameter belongs to the fittest, nullify gradients accordingly.

    Args:
        policy: the actor model
        log_probs: the policy's batch of log probabilities for selected actions
        advantages: advantages corresponding with each log prob
        selection_rate: fraction of parameters that should belong to fittest and get saved
        temp: softmax temperature
        probas: fitness probabilities
        min_adv: minimum advantage threshold
        nonzero: positive fitness only
        uniform_sampling: test ignoring fitness and just sampling uniformly
        deterministic: test selecting top k instead of sampling
        gradient_agg: test
        do_surfn: test
        credit_method: test
        credit_method_2: test
        fitness_dist: test
        probas_dist: test
        select_method: test
    """

    reset(policy)

    if probas is None and do_surfn:
        log_probs.sum().backward(retain_graph=True)
        autograd_hacks.compute_grad1(policy)

        # todo softmax * advantages for fitness
        grads = torch.cat([param.grad1.flatten(start_dim=1) for param in policy.parameters()
                           if hasattr(param, 'grad1')], dim=1)

        param_vals = torch.cat([param.data.flatten() for param in policy.parameters() if hasattr(param, 'grad1')])

        # credit = torch.relu(grads)
        # credit = torch.abs(grads)

        # credit = grads

        # credit_sum = credit.sum(dim=0)
        # advantages = advantages.view([credit.shape[0]] + [1] * (len(credit.shape) - 1))
        # credit_advantages_sum = torch.sum(credit * advantages, dim=0)

        # fitness = torch.sum(torch.softmax(credit, dim=0) * advantages[:, None], dim=0)

        # todo param_vals downstream cha?
        # fitness = torch.sum(torch.softmax(credit, dim=0) * advantages[:, None], dim=0) * torch.abs(param_vals)

        # credit = grads * param_vals[None, :]  # do gs cor
        # credit = grads * torch.sign(param_vals[None, :])
        # credit = torch.sign(grads) * param_vals[None, :]
        # credit = param_vals[None, :] / grads
        # credit = fparam_vals[None, :] / grads
        # credit = s fparam_vals[None, :] / grads
        # credit = -torch.abs(grads)
        # credit = torch.sign(grads) * torch.sign(param_vals[None, :])

        # credit = torch.relu(torch.sign(grads) * torch.sign(param_vals[None, :]))
        # probas = credit / torch.sum(credit, dim=0, keepdim=True)
        # fitness = torch.sum(probas * advantages[:, None], dim=0)

        # credit = torch.relu(grads * param_vals[None, :])
        # probas = credit / torch.sum(credit, dim=0, keepdim=True)
        # fitness = torch.sum(probas * advantages[:, None], dim=0)

        #   fitness = torch.sign(grads) * torch.sign(param_vals[None, :]) * advantages[:, None]
        #   fitness = torch.sum(fitness, dim=0)

        #     credit = torch.sign(grads) * torch.sign(param_vals[None, :])
        #     fitness = credit * advantages[:, None]
        #     fitness = torch.sum(fitness, dim=0) / torch.sum(credit, dim=0)

        #     credit = torch.sign(grads) * param_vals[None, :]  # bt
        #     fitness = credit * advantages[:, None]  # b+b=g
        #     fitness = torch.sum(fitness, dim=0) / torch.sum(credit, dim=0)

        #     credit = torch.sign(grads) * torch.sign(param_vals[None, :])  # bt
        #     fitness = credit * advantages[:, None]  # b+b=g
        #     fitness = torch.sum(fitness, dim=0) * torch.abs(param_vals)

        # credit = torch.sign(grads) * torch.sign(param_vals[None, :])  # s
        # fitness = credit * advantages[:, None]   # b+b=g, bmd
        # fitness = credit * advantages[:, None] / torch.abs(grads)  # b+b=g, bmd
        # fitness = torch.sum(fitness, dim=0) * torch.abs(param_vals)  # bt

        #  fitness = torch.sign(grads) * param_vals[None, :] * advantages[:, None]
        # fitness = torch.sign(grads) * param_vals[None, :] * advantages[:, None] / grads
        # fitness = torch.sum(fitness, dim=0)

        # fitness = grads * param_vals[None, :] * advantages[:, None]
        # fitness = torch.sum(fitness, dim=0)

        #           fitness = credit - torch.min(credit, dim=0, keepdim=True)[0] + 0.0001
        #           fitness = fitness / torch.sum(fitness, dim=0, keepdim=True)
        #           fitness = torch.sum(fitness * advantages[:, None], dim=0)

        # fitness = torch.sum(torch.softmax(credit, dim=0) * advantages[:, None], dim=0)

        # todo inverse fit *, ren   or neg?
        # fitness = credit_advantages_sum / credit_sum
        # fitness = credit_advantages_sum * param_vals / credit_sum
        # todo compatibility with negative rewards?
        # todo -inf ?
        # fitness[torch.isnan(fitness)] = 0

        # bt adv sel
        # credit = torch.relu(torch.sign(grads) * param_vals[None, :])  # s
        # fitness = credit / torch.sum(credit, dim=1, keepdim=True)
        # fitness[torch.isnan(fitness)] = 0
        # fitness = fitness * advantages[:, None]
        # fitness = torch.sum(fitness, dim=0)

        # fitness = torch.softmax(fitness, dim=0) * advantages[:, None]
        # fitness = torch.sum(fitness, dim=0) / torch.sum(credit, dim=0)
        # fitness = torch.sum(fitness, dim=0) * torch.abs(param_vals)  # bt

        # ide  todo / grads/ps/abs
        # fitness = torch.sum(torch.softmax(credit, dim=0) * advantages[:, None], dim=0) * torch.abs(param_vals)
        # fitness = fitness / torch.abs(grads)  # bmd, nav

        if credit_method == "signs":
            credit = torch.sign(grads) * torch.sign(param_vals[None, :])
        elif credit_method == "signs-bt":
            credit = torch.sign(grads) * param_vals[None, :]
        elif credit_method == "signs-gc":
            credit = grads * torch.sign(param_vals[None, :])
        elif credit_method == "bt-gc":
            credit = grads * param_vals[None, :]
        else:
            credit = grads

        if credit_method_2 == "relu":
            credit = torch.relu(credit)
        elif credit_method_2 == "abs":
            credit = torch.abs(credit)
        elif credit_method_2 == "shift-dim":
            credit = credit - torch.min(credit, dim=0, keepdim=True)[0] + 0.001
        elif credit_method_2 == "shift":
            credit = credit - torch.min(credit) + 0.001
        else:
            credit = credit

        if fitness_dist == "sm":
            credit_dist = torch.softmax(credit, dim=1)
            fitness = credit_dist * advantages[:, None]
            fitness = torch.sum(fitness, dim=0)
        elif fitness_dist == "None" or fitness_dist is None:
            fitness = torch.sum(credit * advantages[:, None], dim=0)
        elif fitness_dist == "credit":
            fitness = torch.sum(credit, dim=0)
        else:
            credit_sum = credit.sum(dim=0)
            credit_advantages_sum = torch.sum(credit * advantages[:, None], dim=0)
            fitness = credit_advantages_sum / credit_sum
            fitness[torch.isnan(fitness)] = 0

        if min_adv is not None:
            fitness[fitness < min_adv] = 0

        # todo sep back on ent after cl, momentum,pro temporary/dec, crit, dec-ran, r-ns, det hi

        # todo mult to dirc

        # logits = fitness - min(fitness) + 0.001
        # logits = fitness ** 2
        # probas = logits / torch.sum(logits)

        # credit_sum[torch.isnan(fitness)] = 0
        # logits = param_vals - min(param_vals) + 0.001
        # logits = credit_sum - min(credit_sum) + 0.001
        # probas = logits / torch.sum(logits)

        # logits = fitness
        # probas = torch.softmax(logits / temp, dim=-1)

        if probas_dist == "sm":
            # higher is more random
            probas = torch.softmax(fitness / temp, dim=-1)
        else:
            if probas_dist == "ssm":
                logits = torch.exp(fitness / temp) - 1  # yields infinities
            else:
                logits = fitness - torch.min(fitness) + 0.001
                if probas_dist == "sq":
                    logits = logits ** 2

            assert not torch.isinf(logits).any()

            probas = logits / torch.sum(logits)
            probas[torch.isnan(probas)] = 0

        optimizer.zero_grad()

    with torch.no_grad():
        if do_surfn:
            if uniform_sampling:
                probas = torch.ones_like(probas) / probas.shape[0]

            num_selected = np.inf if selection_rate is None else max(1, int(selection_rate * probas.shape[0]))
            # num_selected = min(torch.count_nonzero(probas).data if min_proba is None
            #                    else probas[probas >= min_proba].shape[0], num_selected)
            if nonzero:
                num_selected = min(torch.count_nonzero(probas).data, num_selected)

            # todo where non-no or min num_sel=min(wh
            if num_selected > 0:
                if deterministic:
                    _, selected = torch.topk(probas, num_selected)
                else:
                    if select_method == "numpy":
                        selected = np.random.choice(a=probas.shape[0], size=num_selected, replace=False, p=probas.numpy())
                    else:
                        rand_state = torch.random.get_rng_state()
                        torch.random.manual_seed(torch.randn(1).data)
                        selected = probas.multinomial(num_samples=num_selected, replacement=False)  # changes seed
                        torch.random.set_rng_state(rand_state)

            is_fittest = torch.zeros(probas.shape[0], dtype=torch.bool)
            if num_selected > 0:
                is_fittest[selected] = True

        handles = []
        fitness_size = 0
        for param in policy.parameters():
            if hasattr(param, 'grad1'):
                if do_surfn:
                    param_fittest = is_fittest[fitness_size:fitness_size + np.prod(param.shape)].view(param.shape)
                handle = param.register_hook(get_survival_hook(param_fittest if do_surfn else None, gradient_agg))
                handles.append(handle)
                fitness_size += np.prod(param.shape)

        policy.__dict__.setdefault('surfn_hooks', []).extend(handles)

        return probas


def get_survival_hook(is_fittest=None, gradient_agg=False):
    """
    Get hook to nullify/freeze gradients for the fittest policy parameters.

    Args:
        is_fittest: boolean vector of whether each policy parameter belongs to the fittest
        gradient_agg: test
    """

    def survival_hook(grad):
        grad = grad.clone()
        if type(gradient_agg) in (int, float) or torch.is_tensor(gradient_agg):
            grad = torch.sign(grad) * torch.abs(gradient_agg)
        elif gradient_agg == "mean":
            grad = torch.sign(grad) * torch.abs(torch.mean(grad))
        elif gradient_agg == "min":
            grad = torch.sign(grad) * torch.min(torch.abs(grad))
        elif gradient_agg == "sign":
            grad = torch.sign(grad)
        if is_fittest is not None:
            grad[is_fittest] = 0
        # grad[not is_fittest] += 1.2
        # don't fe
        # todo c
        # grad =
        # grad[fitness < threshold] = 0
        # grad = torch.sign
        return grad

    return survival_hook


def reset(policy):
    """
    Remove survival hooks.

    Args:
        policy: the actor model
    """
    if hasattr(policy, 'surfn_hooks'):
        for handle in policy.surfn_hooks:
            handle.remove()
        del policy.surfn_hooks
