"""Script used to train agents."""
import argparse
from pathlib import Path
import os
import tonic
from tonic.torch.agents import PPO, TRPO, A2C, SAC


def run(env, alg, agg, seed):
    cur_path = Path(__file__).absolute()
    snapshots_path = Path('./results')
    snapshots_path.mkdir(exist_ok=True)

    is_remote = not Path("/Users/samlerman").exists()
    debugged_logger = True
    if is_remote:
        from clearml import Task
        if debugged_logger:
            from utils import logger

        task = Task.init(project_name="SurF'N", task_name="run", output_uri=str(snapshots_path))

    def train(
            header, agent, environment, trainer, before_training, after_training,
            parallel, sequential, seed, name
    ):
        '''Trains an agent on an environment.'''

        # Capture the arguments to save them, e.g. to play with the trained agent.
        args = dict(locals())

        # Run the header first, e.g. to load an ML framework.
        if header:
            exec(header)

        # Build the agent.
        if isinstance(agent, str):
            agent = eval(agent)

        # Build the train and test environments.
        _environment = environment
        environment = tonic.environments.distribute(
            lambda: eval(_environment), parallel, sequential)
        test_environment = tonic.environments.distribute(
            lambda: eval(_environment))

        # Choose a name for the experiment.
        if hasattr(test_environment, 'name'):
            environment_name = test_environment.name
        else:
            environment_name = test_environment.__class__.__name__
        if not name:
            if hasattr(agent, 'name'):
                name = agent.name
            else:
                name = agent.__class__.__name__
            if parallel != 1 or sequential != 1:
                name += f'-{parallel}x{sequential}'

        # Initialize the logger to save data to the path environment/name/seed.
        path = os.path.join(environment_name, name, str(seed))
        tonic.logger.initialize(path, script_path=cur_path.absolute(), config=args)

        # Build the trainer.
        trainer = eval(trainer)
        trainer.initialize(
            agent=agent, environment=environment,
            test_environment=test_environment, seed=seed)

        # Run some code before training.
        if before_training:
            exec(before_training)

        # Train.
        trainer.run()

        # Run some code after training.
        if after_training:
            exec(after_training)

    if is_remote and debugged_logger:
        logger.initialize(task.get_logger())

    from updaters.actors_grad_agg import ClippedRatioGradAgg, TrustRegionPolicyGradientGradAgg, StochasticPolicyGradientGradAgg, TwinCriticSoftDeterministicPolicyGradientGradAgg

    header = 'import tonic.torch; import sys; sys.path.append("{}")'.format(Path(__file__).parent.absolute())
    if alg == "PPO":
        agent = PPO(actor_updater=ClippedRatioGradAgg(agg=agg))
    elif alg == "TRPO":
        agent = TRPO(actor_updater=TrustRegionPolicyGradientGradAgg(agg))
    elif alg == "A2C":
        agent = A2C(actor_updater=StochasticPolicyGradientGradAgg(agg=agg))
    elif alg == "SAC":
        agent = SAC(actor_updater=TwinCriticSoftDeterministicPolicyGradientGradAgg(agg=agg))
    else:
        assert False
    environment = 'tonic.environments.Bullet("{}")'.format(env)
    trainer = 'tonic.Trainer(epoch_steps=50000, steps=int(5e6))'
    before_training = False
    after_training = False
    parallel = 1
    sequential = 1
    seed = seed
    name = "{}_{}_{}_{}".format(env, alg, agg, seed)
    # name = "PPO_0"
    print(name)

    os.chdir('./results')
    train(header, agent, environment, trainer, before_training, after_training, parallel, sequential, seed, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=str, default='sign')
    parser.add_argument("--alg", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(env=args.env, alg=args.alg, agg=args.agg, seed=args.seed)