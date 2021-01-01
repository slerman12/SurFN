from tonic.torch import agents
from updaters.actors import ClippedRatioSurFN


class SurFNPPO(agents.PPO):
    def __init__(self, model=None, replay=None, actor_updater=None, critic_updater=None):
        actor_updater = actor_updater or ClippedRatioSurFN()
        super().__init__(model=model, replay=replay, actor_updater=actor_updater, critic_updater=critic_updater)

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed)
        self.seed = seed

    def _update(self):
        self.actor_updater.reset(self.seed)
        super(SurFNPPO, self)._update()
