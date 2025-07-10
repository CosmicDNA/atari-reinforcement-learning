#!/bin/sh

python -m rl_zoo3.record_video \
    --algo ppo \
    --env ALE/Breakout-v5 \
    --n-timesteps 5000 \
    --folder logs \
    --load-best