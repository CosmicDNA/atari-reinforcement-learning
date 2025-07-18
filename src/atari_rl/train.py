import sys
from pathlib import Path

from rl_zoo3.train import train as rl_zoo3_train  # type: ignore

from atari_rl import config
from atari_rl.logger import GRAY, NC, logger
from atari_rl.utils import find_experiment_folder, find_model_path


def get_latest_model() -> Path:
    """Finds the latest model file to resume training from."""
    latest_exp_folder = find_experiment_folder(exp_id=None)
    if not latest_exp_folder:
        sys.exit(1)

    # For resuming, we prefer the final model over the best model.
    model_path = find_model_path(latest_exp_folder, prefer_best=False)
    if not model_path:
        logger.error(f"Could not find a model to resume training in {latest_exp_folder}.")
        sys.exit(1)
    return model_path


def run_training(resume_from: Path | None = None):
    """Constructs and runs the training command."""
    args = [
        "--algo",
        config.ALGO,
        "--env",
        config.ENV,
        "--log-folder",
        config.LOG_FOLDER,
        "--vec-env",
        config.VEC_ENV,
        "--n-eval-envs",
        config.N_EVAL_ENVS,
        "--eval-episodes",
        config.EVAL_EPISODES,
        "--eval-freq",
        config.EVAL_FREQ,
        "--save-freq",
        config.SAVE_FREQ,
        "--log-interval",
        config.LOG_INTERVAL,
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

    logger.info(f"Calling rl_zoo3.train with args:\n{GRAY}{' '.join(args)}{NC}")
    try:
        rl_zoo3_train()
    finally:
        sys.argv = original_argv  # Restore original arguments


def start_training(resume: bool):
    """Starts a new training session or resumes an existing one."""
    if resume:
        latest_model = get_latest_model()
        logger.info(f"Resuming training from: {latest_model}")
        run_training(resume_from=latest_model)
    else:
        logger.info("Starting new training session.")
        run_training()
