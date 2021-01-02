"""Script used to train agents."""

import argparse
from pathlib import Path
from trains import Task, Logger
from utils import logger

from tonic.train import train


snapshots_path = Path('./results')
snapshots_path.mkdir(exist_ok=True)


if __name__ == '__main__':
    # Argument parsing.
    task = Task.init(project_name="SurF'N", task_name="trains_plot", output_uri=str(snapshots_path))
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
    logger.initialize(task.get_logger())
    train(**args)
