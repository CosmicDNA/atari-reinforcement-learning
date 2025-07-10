#!/bin/sh
python -m rl_zoo3.train \
    --algo ppo \
    --env ALE/Breakout-v5 --env-kwargs frameskip:1 repeat_action_probability:0 \
    --hyperparams n_envs:8 n_timesteps:100000000 \
    --n-eval-envs 1 \
    --eval-episodes 1 \
    --eval-freq 100000 \
    --eval-env-kwargs frameskip:1 repeat_action_probability:0 \
    --save-freq 1000000 \
    --vec-env=subproc \
    --log-interval 10 \
    --progress