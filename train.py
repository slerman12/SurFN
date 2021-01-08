"""Script used to train agents."""

import argparse
from pathlib import Path
import os

import tonic

from agents.surfn_ppo import SurFNPPO

is_remote = not Path("/Users/samlerman").exists()
if is_remote:
    from clearml import Task
    from utils import logger

    snapshots_path = Path('./results')
    snapshots_path.mkdir(exist_ok=True)

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
    tonic.logger.initialize(path, script_path=__file__, config=args)

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


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def none_or_float(value):
        if value == 'None':
            return None
        return float(value)

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--repeat", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--reiterate", type=str2bool, nargs='?', const=True, default=True)
                        # , default=True)
    parser.add_argument("--adv_run_rate", type=none_or_float, default=None)
    parser.add_argument("--selection_dec_rate", type=none_or_float, default=None)
    parser.add_argument("--selection_rate", type=none_or_float, default=0.04)
    parser.add_argument("--min_adv", type=none_or_float, default=0)
                        # default=None)
    parser.add_argument("--gradient_agg", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--do_surfn", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--uniform_sampling", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--deterministic", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--credit_method", type=none_or_str, default=None)
                        # default="bt-gc")
    parser.add_argument("--credit_method_2", type=none_or_str, default="relu")
                        # default="shift")
    parser.add_argument("--fitness_dist", type=none_or_str, default="sum")
    parser.add_argument("--probas_dist", type=none_or_str, default="sq")
    parser.add_argument("--temp", type=none_or_float, default=-.2)
    parser.add_argument("--nonzero", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--select_method", type=none_or_str, default="numpy")
    parser.add_argument("--env", type=none_or_str, default="AntBulletEnv-v0")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if is_remote:
        logger.initialize(task.get_logger())

    header = 'import tonic.torch; import sys; sys.path.append()'.format(Path(__file__).parent.absolute())
    agent = SurFNPPO(**{key: vars(args)[key] for key in vars(args) if key not in ["env", "seed"]})
    environment = 'tonic.environments.Bullet("{}")'.format(args.env)
    trainer = 'tonic.Trainer(epoch_steps=50000, steps=int(5e6))'
    before_training = False
    after_training = False
    parallel = 1
    sequential = 1
    seed = args.seed
    # name = ""
    # for key in vars(args):
    #     name = "{}_{}_".format(key, vars(args)[key])
    name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(*list(vars(args).values()))

    os.chdir('./results')
    train(header, agent, environment, trainer, before_training, after_training, parallel, sequential, seed, name)
