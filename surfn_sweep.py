import torch
import numpy as np
import autograd_hacks


def set_fittest(policy, advantages, probas=None, log_probs=None, uniform_sampling=False, deterministic=False,
                grad_credit_type="g", grad_credit_op="", weight_credit_op="raw", credit_op="relu", credit_dist_op="",
                adv_credit_type="sum", align_div_by=False, min_adv=0, probas_dist="squared", selection_rate=0.04,
                nonzero=True, select_method="numpy", divide_by_fitness=False, gradient_norm=False, gradient_agg=False):
    """
    Get boolean vector of whether each policy parameter belongs to the fittest, nullify gradients accordingly.

    Args:
        policy: the actor model
        advantages: advantages corresponding with each log prob
        selection_rate: fraction of parameters that should belong to fittest and get saved
        probas: fitness probabilities
        min_adv: minimum advantage threshold
        nonzero: positive fitness only
        uniform_sampling: test ignoring fitness and just sampling uniformly
        deterministic: test selecting top k instead of sampling
        gradient_agg: test
        select_method: test
        log_probs: test
        div_by_align: test
    """
    with torch.no_grad():
        if probas is None:
            reset(policy)
            if grad_credit_type or align_div_by or gradient_norm:
                autograd_hacks.compute_grad1(policy)
                raw_grads = torch.cat([param.grad1.flatten(start_dim=1) for param in policy.parameters()
                                       if hasattr(param, 'grad1')], dim=1)
                adv_rat_grads = -raw_grads
                if gradient_norm:
                    # Note: norm only based on linear and conv layers
                    norm_raw_grads = torch.nn.functional.normalize(raw_grads, p=1, dim=1)
                    norm_raw_grads_avg = torch.mean(norm_raw_grads, dim=0)
            # if gradient_norm:
            #     norm_orig_grads = torch.nn.functional.normalize(torch.cat([torch.flatten(param.grad)
            #                                                                for param in policy.parameters()]), p=1)
            if weight_credit_op:
                weights = torch.cat([param.data.flatten() for param in policy.parameters() if hasattr(param, 'grad1')])

            if grad_credit_type == "arg":
                credit = adv_rat_grads
            elif grad_credit_type == "ag":
                assert log_probs is not None
                adv_grads = adv_rat_grads * torch.exp(log_probs[:, None])
                credit = adv_grads
            elif grad_credit_type == "g":
                assert log_probs is not None
                rat_grads = adv_rat_grads / advantages[:, None]
                grads = rat_grads * torch.exp(log_probs[:, None])
                credit = grads
            else:
                credit = 1

            if not grad_credit_op:
                grad_credit_op = ""
            if "sign" in grad_credit_op:
                credit = torch.sign(credit)
            if "relu" in grad_credit_op:
                credit = torch.relu(credit)
            if "abs" in grad_credit_op:
                credit = torch.abs(credit)
            if "norm" in grad_credit_op:
                credit = torch.nn.functional.normalize(credit, p=1, dim=1)

            if weight_credit_op == "raw":
                credit = credit * weights[None, :]
            if weight_credit_op == "relu":
                credit = credit * torch.relu(weights[None, :])
            if weight_credit_op == "sign":
                credit = credit * torch.sign(weights[None, :])

            if credit_op == "relu":
                credit = torch.relu(credit)
            if credit_op == "sign":
                credit = torch.sign(credit)
            if credit_op == "abs":
                credit = torch.abs(credit)

            if credit_dist_op == "shift-dim":
                credit = credit - torch.min(credit, dim=0, keepdim=True)[0] + 0.001
            if credit_dist_op == "shift":
                credit = credit - torch.min(credit) + 0.001
            if credit_dist_op == "softmax":
                credit = torch.softmax(credit, dim=0)

            if adv_credit_type == "shift-adv":
                fitness = torch.sum(credit * (advantages[:, None] - torch.min(advantages) + 0.001), dim=0)
            elif adv_credit_type == "sum":
                assert credit_op in ["relu", "abs"] or credit_dist_op
                credit_sum = credit.sum(dim=0)
                credit_advantages_sum = torch.sum(credit * advantages[:, None], dim=0)
                fitness = credit_advantages_sum / credit_sum
                fitness[torch.isnan(fitness)] = 0
            elif adv_credit_type == "raw":
                fitness = torch.sum(credit * advantages[:, None], dim=0)
            else:
                fitness = torch.sum(credit, dim=0)

            if align_div_by:
                # Commented out: alignment w.r.t. good/bad actions
                # negative_adv = torch.sum(adv_grads[advantages < 0], dim=0)
                # positive_adv = torch.sum(adv_grads[advantages > 0], dim=0)
                # align = torch.relu(negative_adv * positive_adv) + 1
                # Consider aligning by grad norm rather than sign
                if align_div_by == "norm":
                    by = torch.nn.functional.normalize(raw_grads, p=1, dim=1)
                else:
                    by = torch.sign(raw_grads)
                align = torch.abs(torch.sum(by, dim=0)) + 1
                fitness = fitness / align

            if min_adv is not None:
                fitness[fitness < min_adv] = 0

            logits = fitness
            if probas_dist:
                logits = logits - torch.min(logits) + 0.001
            if not divide_by_fitness:
                if probas_dist == "squared":
                    logits = logits ** 2
                if probas_dist:
                    probas = logits / torch.sum(logits)
                    assert not torch.isinf(probas).any()
                    assert not torch.isnan(probas).any()
            else:
                probas = logits
            # todo sep back on ent after cl, momentum,pro temporary/dec, crit, dec-ran, r-ns, det hi, mult to dirc

            if uniform_sampling:
                probas = torch.ones_like(probas) / probas.shape[0]

        if not divide_by_fitness:
            num_selected = np.inf if selection_rate is None else int(selection_rate * probas.shape[0])
            if nonzero:
                num_selected = min(torch.count_nonzero(probas).data, num_selected)

            if num_selected > 0:
                if deterministic:
                    _, selected = torch.topk(probas, num_selected)
                else:
                    if select_method == "numpy":
                        selected = np.random.choice(a=probas.shape[0], size=num_selected, replace=False, p=probas.numpy())
                    else:
                        # rand_state = torch.random.get_rng_state()
                        # torch.random.seed()
                        selected = probas.multinomial(num_samples=num_selected, replacement=False)  # changes seed
                        # torch.random.set_rng_state(rand_state)

            is_fittest = torch.zeros(probas.shape[0], dtype=torch.bool)
            if num_selected > 0:
                is_fittest[selected] = True

        handles = []
        fitness_size = 0
        # orig_grad_size = 0
        for param in policy.parameters():
            if hasattr(param, 'grad1'):
                # if gradient_norm:
                #     grad = norm_orig_grads[orig_grad_size:orig_grad_size + np.prod(param.shape)].view(param.shape)
                if gradient_norm:
                    grad = norm_raw_grads_avg[fitness_size:fitness_size + np.prod(param.shape)].view(param.shape)
                else:
                    grad = param.grad
                param_fittest = is_fittest[fitness_size:fitness_size + np.prod(param.shape)].view(param.shape)
                survival_hook = get_survival_hook(param_fittest, gradient_agg, divide_by_fitness)
                param.grad = survival_hook(grad)
                if not gradient_norm:
                    handle = param.register_hook(survival_hook)
                    handles.append(handle)
                fitness_size += np.prod(param.shape)
            # orig_grad_size += np.prod(param.shape)

        if not gradient_norm:
            policy.__dict__.setdefault('surfn_hooks', []).extend(handles)

        return probas


def get_survival_hook(is_fittest=None, gradient_agg=False, divide_by_fitness=False):
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
            if divide_by_fitness:
                grad = grad / is_fittest
            else:
                grad[is_fittest] = 0
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


