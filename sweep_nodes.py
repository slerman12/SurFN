import random
import multiprocessing
from train_nodes import run
import itertools

envs = ["AntBulletEnv-v0", "HalfCheetahBulletEnv-vo", "HopperBulletEnv-v0", "ReacherBulletEnv-v0",
        "Walker2DBulletEnv-v0", "InvertedDoublePendulumBulletEnv-v0"]
algs = ["PPO", "A2C"]
aggs = ["sign", None]
seeds = [random.randint(0, 200), random.randint(200, 400), random.randint(400, 600), random.randint(600, 800), random.randint(800, 1000)]

args = list(itertools.product(*[envs, algs, aggs, seeds]))

num_cpus = multiprocessing.cpu_count()
print(num_cpus)

with multiprocessing.Pool(num_cpus) as processing_pool:
    processing_pool.map(run, args)

