import torch
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.ppo import PPO
from mushroom_rl.utils.minibatches import minibatch_generator
import surfn
import autograd_hacks


class SurFNPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(SurFNPPO, self).__init__(*args, **kwargs)
        autograd_hacks.add_hooks(self.policy)

    def _update_policy(self, obs, act, adv, old_log_p):
        probas = []
        for epoch in range(self._n_epochs_policy):
            i = 0
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size, obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                new_log_probs = self.policy.log_prob_t(obs_i, act_i)
                probas_i = surfn.set_fittest(self.policy, new_log_probs, adv_i, self._optimizer,
                                             probas=probas[i] if not epoch else None)
                probas.append(probas_i)
                i += 1
                prob_ratio = torch.exp(
                    new_log_probs - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo,
                                            1 + self._eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss.backward()
                self._optimizer.step()