import argparse
import subprocess
import sys
from pathlib import Path

from . import config


def get_latest_model() -> Path:
    """Finds the latest model file to resume training from."""
    env_path_name = config.ENV.replace("/", "-")
    log_path = Path(config.LOG_FOLDER)

    # Find all experiment directories for the given algo and env
    exp_dirs = list(log_path.glob(f"{config.ALGO}/{env_path_name}_*"))
    if not exp_dirs:
        print(f"\033[0;31mError:\033[0m No experiment folder found for '{config.ALGO}' on '{config.ENV}' in '{config.LOG_FOLDER}'.", file=sys.stderr)
        print("Please run a training session first with 'python scripts/train.py'.", file=sys.stderr)
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
    command = [
        sys.executable,  # Use the same python interpreter that is running this script
        "-m", "rl_zoo3.train",
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
    command.extend(["--env-kwargs"] + config.ENV_KWARGS)
    command.extend(["--eval-env-kwargs"] + config.ENV_KWARGS)
    command.extend(["--hyperparams"] + config.HYPERPARAMS)

    if resume_from:
        command.extend(["--trained-agent", str(resume_from)])

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    """Entry point for the atari-train command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    args = parser.parse_args()

    if args.resume:
        latest_model = get_latest_model()
        print(f"Resuming training from: {latest_model}")
        run_training(resume_from=latest_model)
    else:
        print("Starting new training session.")
        run_training()


if __name__ == "__main__":
    main()