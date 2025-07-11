# Atari Reinforcement Learning with Stable-Baselines3 & RL Zoo

This project provides a streamlined setup for training and evaluating reinforcement learning agents on Atari 2600 games. It is based on the workflow demonstrated in this [YouTube tutorial](https://www.youtube.com/watch?v=aQsaH7Tzvp0&t=329s) and uses a collection of simple shell scripts to manage common tasks like training, evaluation, and video recording.

The core of the project leverages powerful libraries like Stable-Baselines3 and RL-Baselines3-Zoo to do the heavy lifting.


https://github.com/user-attachments/assets/68943893-f0ef-4e58-8dbb-88a6ae038560



## ✨ Key Features

-   **Cross-Platform Workflow:** Simple Python scripts (`train.py`, `enjoy.py` and `record-video.py`) to abstract away complex commands and ensure compatibility across platforms.
-   **Powered by RL Zoo:** Leverages the robust framework of `rl-zoo3` for training, evaluation, and hyperparameter management.
-   **Easy to Customize:** Scripts can be easily modified to train on different Atari games or with different RL algorithms.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+

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
    uv pip install -e .
    ```

4.  **Atari ROMs:**
    This project uses `ale-py` to automatically download and install the necessary Atari ROMs during the dependency installation process.
> [!CAUTION]
> By proceeding with the installation, you are confirming that you have the legal right to use these ROMs.

## 🎮 Usage


### Train a New Agent
To start training an agent from scratch, run the training script. This will save logs and the trained model in the `logs/` directory.
```bash
atari-rl train
```

### Resume Training
If a training session was interrupted, you can resume from the last saved checkpoint.
```bash
atari-rl train --resume
```

### Watch the Agent Play
Once you have a trained model, you can watch it play the game. This script will load the best-performing model from the `logs/` directory.
```bash
atari-rl enjoy # optionally, add “--exp-id n”, where n is the experiment number you want to watch
```

### Record a Video
To save a video of your agent playing, use the recording script. The video will be saved in a `videos/` folder inside the corresponding log directory.
```bash
atari-rl record-video # optionally, add “--exp-id n”, where n is the experiment number you want to watch
```

## 🔧 Customization

All experiment parameters are centralized in `src/atari_rl/config.py`. To change the game, algorithm, or hyperparameters for all scripts at once, simply edit this file.

For example, to train a **DQN** agent on **Pong**, you would modify `src/atari_rl/config.py` like this:
```shell
ALGO="dqn"
ENV="ALE/Pong-v5"
```

## 📁 Project Structure

```plaintext
.
├── src/
│   └── atari_rl/         # Main Python package
│       ├── __init__.py
│       ├── config.py
│       ├── train.py
│       ├── enjoy.py
│       └── record_video.py
├── logs/                 # (Created automatically) Stores trained models and TensorBoard logs
├── .gitignore
├── pyproject.toml        # Project definition and dependencies
└── README.md
```

## 🤝🏿 Acknowledgements

This project stands on the shoulders of the following giants:
- [**RL Baselines3 Zoo**](https://github.com/DLR-RM/rl-baselines3-zoo): A Training Framework for Stable Baselines3 Reinforcement Learning Agents.
- [**Gymnasium**](https://github.com/Farama-Foundation/Gymnasium): A Python library for developing and comparing reinforcement learning algorithms.
- [**PyTorch**](https://pytorch.org/): An open-source machine learning framework that accelerates the path from research prototyping to production deployment.
- [**The Arcade Learning Environment**](https://github.com/Farama-Foundation/Arcade-Learning-Environment): A simple framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.
- [**Moviepy**](https://github.com/Zulko/moviepy): A Python library for video editing: cuts, concatenations, title insertions, video compositing (a.k.a. non-linear editing), video processing, and creation of custom effects.
- [**OpenCV**](https://github.com/opencv/opencv-python): Pre-built CPU-only OpenCV packages for Python.
