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


# def wrapper(env_alg_agg_seed):
#     env, alg, agg, seed = env_alg_agg_seed
#     return run(env=env, alg=alg, agg=agg, seed=seed)
#
#
# with multiprocessing.Pool(num_cpus) as processing_pool:
#     # accumulate results in a dictionary
#     results = processing_pool.map(wrapper, args)

# use starmap and call `run` directly
with multiprocessing.Pool(8) as processing_pool:
    processing_pool.starmap(run, args)

