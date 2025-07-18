import os
from pathlib import Path

from dotenv import load_dotenv

from atari_rl.logger import logger

# --- .env File Template ---
# This template is used to create a .env file if one doesn't exist in the current directory.
ENV_TEMPLATE = """
# --- Atari Reinforcement Learning Configuration ---
# This file contains the shared parameters for all experiment scripts.
# Edit the variables here to change the algorithm, environment, and hyperparameters for this directory.

# The algorithm to use for training and evaluation.
# See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
ALGO="ppo"

# The Gymnasium environment ID for the Atari game.
# A list of Atari environments can be found here: https://gymnasium.farama.org/environments/atari/
ENV="ALE/Breakout-v5"

# The root directory for saving logs and trained models.
LOG_FOLDER="logs"

# Additional arguments for the environment.
# For Atari, `frameskip:1` and `repeat_action_probability:0` are common for better performance.
# This should be a comma-separated list.
ENV_KWARGS="frameskip:1,repeat_action_probability:0"

# Total number of timesteps for training.
N_TIMESTEPS="100000000"

# --- Advanced Training Hyperparameters ---
EVAL_FREQ="1000000"
SAVE_FREQ="10000000"
LOG_INTERVAL="10"
N_EVAL_ENVS="1"
EVAL_EPISODES="1"
VEC_ENV="subproc"

# --- Video Recording Parameters ---
RECORD_TIMESTEPS="50"
"""


def _load_config() -> dict:
    """Loads configuration from a .env file in the current directory.

    If the file doesn't exist, it creates one from the template.
    Returns a dictionary of configuration values.
    """
    env_path = Path(".env")

    if not env_path.exists():
        logger.info(f"No .env file found. Creating a default one in {env_path.resolve()}")
        with env_path.open("w", encoding="utf-8") as f:
            f.write(ENV_TEMPLATE.strip() + "\n")

    # Load the .env file into the environment
    load_dotenv(dotenv_path=env_path)

    # Helper to get value from os.environ or default
    def get_str(key, default):
        return os.environ.get(key, default)

    def get_list(key, default):
        val = os.environ.get(key)
        if val:
            return [item.strip() for item in val.split(",") if item.strip()]
        return default

    # Load values, parsing them into correct types
    config = {
        "ALGO": get_str("ALGO", "ppo"),
        "ENV": get_str("ENV", "ALE/Breakout-v5"),
        "LOG_FOLDER": get_str("LOG_FOLDER", "logs"),
        "ENV_KWARGS": get_list("ENV_KWARGS", ["frameskip:1", "repeat_action_probability:0"]),
        "N_TIMESTEPS": get_str("N_TIMESTEPS", "100000000"),
        "EVAL_FREQ": get_str("EVAL_FREQ", "1000000"),
        "SAVE_FREQ": get_str("SAVE_FREQ", "10000000"),
        "LOG_INTERVAL": get_str("LOG_INTERVAL", "10"),
        "N_EVAL_ENVS": get_str("N_EVAL_ENVS", "1"),
        "EVAL_EPISODES": get_str("EVAL_EPISODES", "1"),
        "VEC_ENV": get_str("VEC_ENV", "subproc"),
        "RECORD_TIMESTEPS": get_str("RECORD_TIMESTEPS", "50"),
    }

    # Special handling for n_envs and constructing HYPERPARAMS
    n_envs = os.cpu_count()
    config["HYPERPARAMS"] = [f"n_envs:{n_envs}", f"n_timesteps:{config['N_TIMESTEPS']}"]

    return config


# --- Load Configuration and Expose as Module Variables ---
_config_data = _load_config()

ALGO = _config_data["ALGO"]
ENV = _config_data["ENV"]
LOG_FOLDER = _config_data["LOG_FOLDER"]
ENV_KWARGS = _config_data["ENV_KWARGS"]
HYPERPARAMS = _config_data["HYPERPARAMS"]
EVAL_FREQ = _config_data["EVAL_FREQ"]
SAVE_FREQ = _config_data["SAVE_FREQ"]
LOG_INTERVAL = _config_data["LOG_INTERVAL"]
N_EVAL_ENVS = _config_data["N_EVAL_ENVS"]
EVAL_EPISODES = _config_data["EVAL_EPISODES"]
VEC_ENV = _config_data["VEC_ENV"]
RECORD_TIMESTEPS = _config_data["RECORD_TIMESTEPS"]
