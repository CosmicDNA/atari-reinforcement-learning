import sys

from rl_zoo3.enjoy import enjoy as rl_zoo3_enjoy  # type: ignore

from atari_rl import config
from atari_rl.logger import GRAY, NC, logger
from atari_rl.utils import find_model_for_evaluation


def watch_agent(exp_id: int | None = None):
    """Constructs and runs the enjoy command for a given experiment ID."""
    model_path, _ = find_model_for_evaluation(exp_id)
    if not model_path:
        sys.exit(1)

    logger.info(f"Loading model from: {model_path}")

    args_list = [
        "--algo",
        config.ALGO,
        "--env",
        config.ENV,
        "--folder",
        config.LOG_FOLDER,  # Still needed for hyperparams
        "--trained-agent",
        str(model_path),
    ]
    args_list.extend(["--env-kwargs"] + config.ENV_KWARGS)

    original_argv = sys.argv
    sys.argv = ["enjoy.py"] + args_list

    logger.info(f"Calling rl_zoo3.enjoy with args:\n{GRAY}{' '.join(args_list)}{NC}")
    try:
        rl_zoo3_enjoy()
    finally:
        sys.argv = original_argv
