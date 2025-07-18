import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from rl_zoo3.train import ALGOS
from rl_zoo3.utils import get_saved_hyperparams
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, NoopResetEnv
from tqdm import tqdm

from atari_rl import config, svg_generation
from atari_rl.logger import logger
from atari_rl.utils import find_model_for_evaluation


def _generate_hq_frames(model_path: Path) -> list[np.ndarray]:
    """Generates a high-quality, 60 FPS list of frames of the agent's gameplay.

    This is the single source of truth for recording the same gameplay across multiple formats.
    """
    # 1. Load the model to determine its expected observation shape (n_stack, height, width).
    algo_class = ALGOS[config.ALGO]
    model = algo_class.load(model_path)
    n_stack = model.observation_space.shape[0]
    obs_height, obs_width = model.observation_space.shape[1], model.observation_space.shape[2]

    # 2. Create a base, non-vectorized environment with frameskip=1 to render every frame.
    hyperparams, _ = get_saved_hyperparams(model_path)
    env_kwargs = hyperparams.get("env_kwargs", {})
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["frameskip"] = 1  # Override any other frameskip settings for recording.
    env = gym.make(config.ENV, **env_kwargs)

    # 3. Manually apply the essential Atari wrappers for game logic.
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    # 4. Set up a manual stack for the agent's observations.
    stacked_frames = np.zeros((n_stack, obs_height, obs_width), dtype=np.uint8)

    logger.info(f"Collecting frames for a {config.RECORD_TIMESTEPS}-step high-quality recording...")
    frames = []

    # Reset environment and prepare initial state
    obs, info = env.reset()  # obs is a color frame
    # Capture the very first frame of the gameplay session.
    # We use .copy() to ensure we save a unique snapshot of the frame, not a reference to the env's internal buffer.
    frames.append(obs.copy())

    # Process the initial observation for the agent's stack
    processed_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    processed_obs = cv2.resize(processed_obs, (obs_width, obs_height), interpolation=cv2.INTER_AREA)
    # Fill the stack with the first frame
    for i in range(n_stack):
        stacked_frames[i] = processed_obs

    try:
        # The agent makes decisions at a lower FPS (e.g., 15 FPS with frameskip=4).
        # We simulate this while rendering every single game frame (60 FPS).
        for _ in tqdm(range(int(config.RECORD_TIMESTEPS)), desc="Agent Steps"):
            # Get action from the agent based on the stacked observations
            # Add a batch dimension (n_env=1) for the model
            action, _ = model.predict(stacked_frames[None, ...], deterministic=True)

            # Simulate frameskip by repeating the action for 4 steps.
            for _ in range(4):
                obs, _, terminated, truncated, info = env.step(action[0])

                # The observation `obs` returned by `env.step()` is the canonical RGB frame
                # when `render_mode="rgb_array"`. Using it directly ensures the recorded frame
                # is exactly what the agent's next decision will be based on (after processing).
                frames.append(obs.copy())

                # Process the observation for the agent's next decision
                processed_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                processed_obs = cv2.resize(processed_obs, (obs_width, obs_height), interpolation=cv2.INTER_AREA)

                # Update the stack by rolling and adding the new frame
                stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
                stacked_frames[-1] = processed_obs

                done = terminated or truncated

                if done:
                    break  # Break inner loop if episode ends mid-frameskip

            if done:
                obs, info = env.reset()
                # Capture the first frame of the new episode after a reset.
                frames.append(obs.copy())
                processed_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                processed_obs = cv2.resize(processed_obs, (obs_width, obs_height), interpolation=cv2.INTER_AREA)
                # Reset the stack with the new frame
                for i in range(n_stack):
                    stacked_frames[i] = processed_obs

    finally:
        env.close()

    return frames


def _save_to_mp4(frames: list[np.ndarray], video_path: Path):
    """Saves a list of frames as a high-quality MP4 video using OpenCV."""
    logger.info(f"Saving to MP4 using OpenCV: {video_path}")
    if not frames:
        logger.warning("No frames to save for MP4.")
        return

    height, width, _ = frames[0].shape
    # Define the codec and create VideoWriter object. 'mp4v' is a good default for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 60.0, (width, height))

    for frame in tqdm(frames, desc="Writing MP4"):
        # OpenCV expects frames in BGR format, but they are rendered in RGB.
        # We must convert the color space before writing.
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    logger.info(f"MP4 video saved successfully to {video_path}")


def create_video(exp_id: int | None, video_format: str):
    """Constructs and runs the video recording command for the specified format."""
    env_path_name = config.ENV.replace("/", "-")

    model_path, exp_folder = find_model_for_evaluation(exp_id)
    if not model_path or not exp_folder:
        sys.exit(1)

    log_path = exp_folder  # The log path is the experiment folder itself.

    logger.info(f"Loading model from: {model_path}")

    # Generate high-quality frames. This is now the single source of truth for all formats.
    frames = _generate_hq_frames(model_path)
    if not frames:
        logger.error("No frames were recorded.")
        sys.exit(1)

    # The original rl_zoo3 script saves videos to a 'videos' subfolder within the experiment's log directory.
    video_folder = log_path / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)
    base_filename = f"replay_{config.ALGO}_{env_path_name}_steps_{config.RECORD_TIMESTEPS}"

    if video_format in ["svg", "all"]:
        svg_path = video_folder / f"{base_filename}.svg"
        svg_generation.save_to_svg(frames, svg_path)

    if video_format in ["mp4", "all"]:
        mp4_path = video_folder / f"{base_filename}.mp4"
        _save_to_mp4(frames, mp4_path)
