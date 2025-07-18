import sys

from rl_zoo3.enjoy import enjoy as rl_zoo3_enjoy  # type: ignore

from atari_rl import config
from atari_rl.logger import GRAY, NC, logger
from atari_rl.utils import find_model_for_evaluation


def watch_agent(exp_id: int | None = None):
    """Constructs and runs the enjoy command for a given experiment ID."""
    model_path, exp_folder = find_model_for_evaluation(exp_id)
    if not model_path or not exp_folder:
        sys.exit(1)

    # Extract the experiment ID from the folder name (e.g., 'ALE-Breakout-v5_10' -> 10)
    try:
        found_exp_id = int(exp_folder.name.split("_")[-1])
    except (ValueError, IndexError):
        logger.error(f"Could not parse experiment ID from folder name: {exp_folder.name}")
        sys.exit(1)

    logger.info(f"Loading model from: {model_path}")

    args_list = [
        "--algo",
        config.ALGO,
        "--env",
        config.ENV,
        "--folder",
        config.LOG_FOLDER,
        "--exp-id",
        str(found_exp_id),
    ]
    # rl-zoo3 enjoy loads the final model by default.
    # We only add --load-best if the model we found is indeed the best one.
    if model_path.name == "best_model.zip":
        args_list.append("--load-best")

    args_list.extend(["--env-kwargs"] + config.ENV_KWARGS)

    original_argv = sys.argv
    sys.argv = ["enjoy.py"] + args_list

    logger.info(f"Calling rl_zoo3.enjoy with args:\n{GRAY}{' '.join(args_list)}{NC}")
    try:
        rl_zoo3_enjoy()
    finally:
        sys.argv = original_argv
