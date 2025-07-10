#!/bin/sh
python -m rl_zoo3.enjoy \
    --algo ppo \
    --env ALE/Breakout-v5 \
    --folder logs \
    --exp-id 9 \
    --load-best