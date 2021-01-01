#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Quick distributed training.
python3 -m tonic.train \
    --header "import tonic.torch" \
    --agent "tonic.torch.agents.PPO(replay=tonic.replays.Segment(size=10, batch_size=2000, batch_iterations=30))" \
    --environment "tonic.environments.Gym('LunarLanderContinuous-v2')" \
    --trainer "tonic.Trainer(epoch_steps=100, steps=500000, save_steps=500000)" \
    --parallel 10 \
    --sequential 100 \
    --name "PPO-torch-demo" \
    --seed 0

mkdir results
cd results || exit
python3 -m tonic.train --header "import tonic.torch; import sys; sys.path.append('/Users/samlerman/Code/SurF\'N'); from updaters.actors import StochasticPolicyGradientSurFN" --agent 'tonic.torch.agents.A2C(actor_updater=StochasticPolicyGradientSurFN())' --environment 'tonic.environments.Bullet("AntBulletEnv-v0")' --trainer 'tonic.Trainer(epoch_steps=50000)' --name SurFN-A2C-XX2 --seed 0
python3 -m tonic.train --header "import tonic.torch" --agent 'tonic.torch.agents.A2C()' --environment 'tonic.environments.Bullet("InvertedPendulumBulletEnv-v0")' --trainer 'tonic.Trainer(epoch_steps=50000)' --name A2C-0 --seed 0
python3 -m tonic.plot --path AntBulletEnv-v0 --baselines all

python3 -m tonic.train --header "import tonic.torch; import sys; sys.path.append('/Users/samlerman/Code/SurF\'N'); from agents.surfn_ppo import SurFNPPO" --agent 'SurFNPPO()' --environment 'tonic.environments.Bullet("AntBulletEnv-v0")' --trainer 'tonic.Trainer(epoch_steps=50000)' --name SurFN-PPO-0 --seed 0

# Plot and reload.
python3 -m tonic.plot --path LunarLanderContinuous-v2 --baselines all &
python3 -m tonic.play --path LunarLanderContinuous-v2/PPO-torch-demo/0 &
wait
