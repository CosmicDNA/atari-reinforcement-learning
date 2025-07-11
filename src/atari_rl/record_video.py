import subprocess
import sys

from . import config

# ANSI color codes
GRAY = '\033[0;90m'
NC = '\033[0m'  # No Color

def create_video(exp_id: int | None):
    """Constructs and runs the record_video command."""
    command = [
        sys.executable,
        "-m", "rl_zoo3.record_video",
        "--algo", config.ALGO,
        "--env", config.ENV,
        "--folder", config.LOG_FOLDER,
        "--load-best",
        "--n-timesteps", config.RECORD_TIMESTEPS,
    ]
    command.extend(["--env-kwargs"] + config.ENV_KWARGS)

    if exp_id is not None:
        command.extend(["--exp-id", str(exp_id)])

    print(f"Running command:\n{GRAY}{' '.join(command)}{NC}")
    subprocess.run(command, check=True)