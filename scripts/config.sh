#!/bin/sh

# --- Main Configuration ---
# This file contains the shared parameters for all experiment scripts.
# Edit the variables here to change the algorithm, environment, and hyperparameters.

# The algorithm to use for training and evaluation.
# See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
ALGO="ppo"

# The Gymnasium environment ID for the Atari game.
# A list of Atari environments can be found here: https://gymnasium.farama.org/environments/atari/
ENV="ALE/Breakout-v5"

# The root directory for saving logs and trained models.
LOG_FOLDER="logs"

# --- Environment Kwargs ---
# Additional arguments for the environment.
# For Atari, `frameskip:1` and `repeat_action_probability:0` are common for better performance.
ENV_KWARGS="frameskip:1 repeat_action_probability:0"

# --- Training Hyperparameters ---
# `n_envs`: Number of parallel environments to use for training.
# `n_timesteps`: Total number of timesteps for training.
HYPERPARAMS="n_envs:8 n_timesteps:100000000"

EVAL_FREQ="1000000"
SAVE_FREQ="10000000"
LOG_INTERVAL="10"
N_EVAL_ENVS="1"
EVAL_EPISODES="1"
VEC_ENV="subproc"

# --- Video Recording Parameters ---
RECORD_TIMESTEPS="5000"

# --- Reusable Functions ---

# Encapsulates the base training command.
# Accepts additional arguments, like `--load-last-checkpoint`.
run_training() {
    python -m rl_zoo3.train \
        --algo "$ALGO" \
        --env "$ENV" --env-kwargs $ENV_KWARGS \
        --log-folder "$LOG_FOLDER" \
        --hyperparams $HYPERPARAMS \
        --n-eval-envs "$N_EVAL_ENVS" \
        --eval-episodes "$EVAL_EPISODES" \
        --eval-freq "$EVAL_FREQ" \
        --eval-env-kwargs $ENV_KWARGS \
        --save-freq "$SAVE_FREQ" \
        --vec-env="$VEC_ENV" \
        --log-interval "$LOG_INTERVAL" \
        --progress \
        "$@" # Pass along any extra arguments to the python script
}

ENJOY_RECORD_PARAMS="--algo "$ALGO" --env "$ENV" --folder "$LOG_FOLDER" --load-best --env-kwargs $ENV_KWARGS"

run_enjoy() {
    python -m rl_zoo3.enjoy \
        $ENJOY_RECORD_PARAMS "$@"
}

run_record_video() {
    python -m rl_zoo3.record_video \
        $ENJOY_RECORD_PARAMS --n-timesteps "$RECORD_TIMESTEPS" "$@"
}
