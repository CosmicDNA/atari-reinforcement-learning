import argparse
import sys

from . import config
from rl_zoo3.enjoy import enjoy as rl_zoo3_enjoy # type: ignore

# ANSI color codes
GRAY = '\033[0;90m'
NC = '\033[0m' # No Color

def main():
    """Constructs and runs the enjoy command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, help="Experiment ID to enjoy.")
    args = parser.parse_args()

    args_list = [
        "--algo", config.ALGO,
        "--env", config.ENV,
        "--folder", config.LOG_FOLDER,
        "--load-best",
    ]
    args_list.extend(["--env-kwargs"] + config.ENV_KWARGS)

    if args.exp_id is not None:
        args_list.extend(["--exp-id", str(args.exp_id)])

    original_argv = sys.argv
    sys.argv = ["enjoy.py"] + args_list

    print(f"Calling rl_zoo3.enjoy with args:\n{GRAY}{' '.join(args_list)}{NC}")
    try:
        rl_zoo3_enjoy()
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()