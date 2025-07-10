# Atari Reinforcement Learning with Stable-Baselines3 & RL Zoo

This project provides a streamlined setup for training and evaluating reinforcement learning agents on Atari 2600 games. It is based on the workflow demonstrated in this [YouTube tutorial](https://www.youtube.com/watch?v=aQsaH7Tzvp0&t=329s) and uses a collection of simple shell scripts to manage common tasks like training, evaluation, and video recording.

The core of the project leverages powerful libraries like Stable-Baselines3 and RL-Baselines3-Zoo to do the heavy lifting.

## ✨ Key Features

-   **Script-based Workflow:** Simple shell scripts (`train.sh`, `enjoy.sh`, `record-video.sh` and `resume-training.sh`) to abstract away complex commands.
-   **Powered by RL Zoo:** Leverages the robust framework of `rl-zoo3` for training, evaluation, and hyperparameter management.
-   **Reproducible Environment:** Uses `uv` for fast and consistent dependency installation.
-   **Easy to Customize:** Scripts can be easily modified to train on different Atari games or with different RL algorithms.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   A Unix-like shell (e.g., bash, zsh) for running the scripts.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CosmicDNA/atari-reinforcement-learning.git
    cd atari-reinforcement-learning
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project uses `uv` for fast dependency management.
    ```bash
    pip install uv
    uv pip install -r requirements/requirements-lock.txt
    ```
> [!NOTE] If you modify `requirements/requirements.in`, you can re-compile the `requirements-lock.txt` file with:
> `uv pip compile --constraint requirements/constraints.txt requirements/requirements.in -o requirements/requirements-lock.txt`

4.  **Atari ROMs:**
    This project uses `ale-py` to automatically download and install the necessary Atari ROMs during the dependency installation process.
> [!CAUTION]
> By proceeding with the installation, you are confirming that you have the legal right to use these ROMs.

## 🎮 Usage

This project uses simple shell scripts located in the `scripts/` directory to handle various tasks. All scripts are pre-configured for the `ALE/Breakout-v5` environment.

> [!TIP]
> Before running the scripts for the first time, you will need to make them executable. You can do this for all scripts at once with the following command:
> ```bash
> chmod +x scripts/*.sh
> ```

### Train a New Agent
To start training an agent from scratch, run the training script. This will save logs and the trained model in the `logs/` directory.
```bash
./scripts/train.sh
```

### Resume Training
If a training session was interrupted, you can resume from the last saved checkpoint.
```bash
./scripts/resume-training.sh
```

### Watch the Agent Play
Once you have a trained model, you can watch it play the game. This script will load the best-performing model from the `logs/` directory.
```bash
./scripts/enjoy.sh
```

### Record a Video
To save a video of your agent playing, use the recording script. The video will be saved in a `videos/` folder inside the corresponding log directory.
```bash
./scripts/record-video.sh
```

## 🔧 Customization

The shell scripts in the `scripts/` folder are your main entry points. You can easily change the game or algorithm by editing the arguments within these files. For example, to train a **DQN** agent on **Pong**, you would edit `scripts/train.sh` to change the `--algo` and `--env` flags.

## 📁 Project Structure

```
.
├── requirements/         # Python dependency files (`.in`, `.txt`, `constraints.txt`)
├── scripts/              # Shell scripts for managing experiments (train, enjoy, etc.)
├── logs/                 # (Created automatically) Stores trained models and TensorBoard logs
├── .gitignore            # Specifies files to be ignored by Git
└── README.md             # This file
```