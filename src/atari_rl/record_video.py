import argparse
import sys

from . import config
from rl_zoo3.record_video import record_video as rl_zoo3_record_video # type: ignore


def main():
    """Constructs and runs the record_video command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, help="Experiment ID to record.")
    args = parser.parse_args()

    args_list = [
        "--algo", config.ALGO,
        "--env", config.ENV,
        "--folder", config.LOG_FOLDER,
        "--load-best",
        "--n-timesteps", config.RECORD_TIMESTEPS,
    ]
    args_list.extend(["--env-kwargs"] + config.ENV_KWARGS)

    if args.exp_id is not None:
        args_list.extend(["--exp-id", str(args.exp_id)])

    original_argv = sys.argv
    sys.argv = ["record_video.py"] + args_list

    print(f"Calling rl_zoo3.record_video with args: {' '.join(args_list)}")
    try:
        rl_zoo3_record_video()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()