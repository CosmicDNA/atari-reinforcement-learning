import argparse
import subprocess
import sys

from . import config


def main():
    """Constructs and runs the enjoy command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=int, help="Experiment ID to enjoy.")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m", "rl_zoo3.enjoy",
        "--algo", config.ALGO,
        "--env", config.ENV,
        "--folder", config.LOG_FOLDER,
        "--load-best",
    ]
    command.extend(["--env-kwargs"] + config.ENV_KWARGS)

    if args.exp_id is not None:
        command.extend(["--exp-id", str(args.exp_id)])

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()