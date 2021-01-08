"""Script used to train agents."""

import argparse
from pathlib import Path
import os

from tonic.train import train

is_remote = not Path("/Users/samlerman").exists()
if is_remote:
    from clearml import Task
    from utils import logger

    snapshots_path = Path('./results')
    snapshots_path.mkdir(exist_ok=True)

    task = Task.init(project_name="SurF'N", task_name="run", output_uri=str(snapshots_path))


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--header')
    parser.add_argument('--agent', required=True)
    parser.add_argument('--environment', '--env', required=True)
    parser.add_argument('--trainer', default='tonic.Trainer()')
    parser.add_argument('--before_training')
    parser.add_argument('--after_training')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--sequential', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--name')
    args = vars(parser.parse_args())
    if is_remote:
        logger.initialize(task.get_logger())
    os.chdir('./results')
    train(**args)
