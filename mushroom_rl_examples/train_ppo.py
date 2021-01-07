import time

from surfn_ppo import SurFNPPO
from mushroom_rl.policy import GaussianTorchPolicy

from mushroom_rl_benchmark import BenchmarkExperiment, BenchmarkLogger
from mushroom_rl_benchmark.builders import EnvironmentBuilder, PPOBuilder
from mushroom_rl_benchmark.builders.network import TRPONetwork as Network


class SurFNPPOBuilder(PPOBuilder):
    """
    AgentBuilder for SurF'N Proximal Policy Optimization algorithm
    """

    def build(self, mdp_info):
        policy = GaussianTorchPolicy(
            Network,
            mdp_info.observation_space.shape,
            mdp_info.action_space.shape,
            **self.policy_params)
        self.critic_params["input_shape"] = mdp_info.observation_space.shape
        self.alg_params['critic_params'] = self.critic_params
        self.alg_params['actor_optimizer'] = self.actor_optimizer
        return SurFNPPO(mdp_info, policy, **self.alg_params)


if __name__ == '__main__':

    logger = BenchmarkLogger(
        log_dir='./logs',
        log_id='ppo_pendulum'
    )

    agent_builder = SurFNPPOBuilder.default(
        actor_lr=3e-4,
        critic_lr=3e-4,
        n_features=32
    )

    env_name = 'PyBullet'
    env_params = dict(
        env_id='AntBulletEnv-v0',
        horizon=200,
        gamma=.99
    )

    env_builder = EnvironmentBuilder(env_name, env_params)
    logger.info('Environment is imported')

    exp = BenchmarkExperiment(agent_builder, env_builder, logger)
    logger.info('BenchmarkExperiment was built successfully')

    start_time = time.time()
    exp.run(
        exec_type='parallel',
        n_runs=10,
        n_epochs=100,
        n_steps=30000,
        n_episodes_test=5,
        max_concurrent_runs=10
    )
    end_time = time.time()
    logger.info('Execution time: {} SEC'.format(end_time-start_time))

    exp.save_plot()
    #exp.show_report()
