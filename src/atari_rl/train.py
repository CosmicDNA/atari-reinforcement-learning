import sys
from pathlib import Path

from . import config

from rl_zoo3.train import train as rl_zoo3_train # type: ignore

# ANSI color codes
GRAY = '\033[0;90m'
NC = '\033[0m'  # No Color

def get_latest_model() -> Path:
    """Finds the latest model file to resume training from."""
    env_path_name = config.ENV.replace("/", "-")
    log_path = Path(config.LOG_FOLDER)

    # Find all experiment directories for the given algo and env
    exp_dirs = list(log_path.glob(f"{config.ALGO}/{env_path_name}_*"))
    if not exp_dirs:
        print(f"\033[0;31mError:\033[0m No experiment folder found for '{config.ALGO}' on '{config.ENV}' in '{config.LOG_FOLDER}'.", file=sys.stderr)
        print("Please run a training session first with 'atari-rl train'.", file=sys.stderr)
        sys.exit(1)

    # Sort directories by experiment number to find the latest
    latest_exp_dir = max(exp_dirs, key=lambda p: int(p.name.split("_")[-1]))

    # The model saved at the end of training is named after the environment.
    final_model_path = latest_exp_dir / f"{env_path_name}.zip"
    best_model_path = latest_exp_dir / "best_model.zip"

    if final_model_path.is_file():
        return final_model_path
    elif best_model_path.is_file():
        return best_model_path
    else:
        print(f"\033[0;31mError:\033[0m No model file found in '{latest_exp_dir}' to resume training.", file=sys.stderr)
        sys.exit(1)


def run_training(resume_from: Path | None = None):
    """Constructs and runs the training command."""
    args = [
        "--algo", config.ALGO,
        "--env", config.ENV,
        "--log-folder", config.LOG_FOLDER,
        "--vec-env", config.VEC_ENV,
        "--n-eval-envs", config.N_EVAL_ENVS,
        "--eval-episodes", config.EVAL_EPISODES,
        "--eval-freq", config.EVAL_FREQ,
        "--save-freq", config.SAVE_FREQ,
        "--log-interval", config.LOG_INTERVAL,
        "--progress",
    ]

    # Add list-based arguments
    args.extend(["--env-kwargs"] + config.ENV_KWARGS)
    args.extend(["--eval-env-kwargs"] + config.ENV_KWARGS)
    args.extend(["--hyperparams"] + config.HYPERPARAMS)

    if resume_from:
        args.extend(["--trained-agent", str(resume_from)])

    # Temporarily replace sys.argv to pass arguments to the training function
    original_argv = sys.argv
    sys.argv = ["train.py"] + args

    print(f"Calling rl_zoo3.train with args:\n{GRAY}{' '.join(args)}{NC}")
    try:
        rl_zoo3_train()
    finally:
        sys.argv = original_argv  # Restore original arguments

def start_training(resume: bool):
    """Starts a new training session or resumes an existing one."""
    if resume:
        latest_model = get_latest_model()
        print(f"Resuming training from: {latest_model}")
        run_training(resume_from=latest_model)
    else:
        print("Starting new training session.")
        run_training()