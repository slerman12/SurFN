import random
import multiprocessing
import itertools
import os
import time

from train_nodes import run


if __name__ == '__main__':
    envs = ["AntBulletEnv-v0", "HalfCheetahBulletEnv-v0", "Walker2DBulletEnv-v0",
            # ]
            "ReacherBulletEnv-v0", "HopperBulletEnv-v0", "InvertedDoublePendulumBulletEnv-v0"]
    # algs = ["PPO",
    #         ]
            # "A2C"]
    # aggs = ["sign",
    #         # ]
    #         None]
    seeds = [143,
             233,
             # ]
            547, 719, 863]

    args = list(itertools.product(*[envs, seeds]))

    num_cpus = multiprocessing.cpu_count()
    print(num_cpus)

    # os.chdir("/Users/samlerman/Code/SurF'N/results")
    # def wrapper(env_alg_agg_seed):
    #     env, alg, agg, seed = env_alg_agg_seed
    #     return run(env=env, alg=alg, agg=agg, seed=seed)
    #
    #
    # with multiprocessing.Pool(num_cpus) as processing_pool:
    #     # accumulate results in a dictionary
    #     results = processing_pool.map(wrapper, args)

    # use starmap and call `run` directly
    # with multiprocessing.Pool(num_cpus) as processing_pool:
    #     processing_pool.starmap(run, args)

    def go(arg):
        os.system('python3 ../sbatch.py --params "--env {} --seed {}" --cpu --name "{}_{}_g_relu"'.format(*arg, *arg))
    #
    #
    for arg in args:
        go(arg)
        time.sleep(3)

    # for arg in args:
    #     os.system('python3 ../train_nodes.py --env {} --alg {} --agg {} --seed {}'.format(*arg))

