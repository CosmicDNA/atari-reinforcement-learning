from pathlib import Path
import re
import sys
import cv2
import gymnasium as gym
import numpy as np
import vtracer
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, NoopResetEnv
from tqdm import tqdm

from rl_zoo3.train import ALGOS
from rl_zoo3.utils import get_model_path, get_saved_hyperparams
from . import config

# ANSI color codes
GRAY = '\033[0;90m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


def _generate_hq_frames(model_path: Path) -> list[np.ndarray]:
    """
    Generates a high-quality, 60 FPS list of frames of the agent's gameplay.
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

    print(f"Collecting frames for a {config.RECORD_TIMESTEPS}-step high-quality recording...")
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


def _save_to_svg(frames: list[np.ndarray], video_path: Path):
    """Saves a list of frames as a differentially rendered, animated SVG."""
    print(f"Saving to SVG using differential rendering: {video_path}")
    if not frames:
        print("Warning: No frames to save for SVG.", file=sys.stderr)
        return

    height, width, _ = frames[0].shape
    fps = 60.0
    delay_s = 1.0 / fps

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        # Use a clip path to prevent any potential overflow from vectorization artifacts.
        f'<defs><clipPath id="clip"><rect x="0" y="0" width="{width}" height="{height}"/></clipPath></defs>',
        # Apply the clip path to a main container group.
        '<g style="clip-path: url(#clip);">',
    ]

    # --- Vectorize and add the first frame (base frame) ---
    print("Vectorizing base frame...")
    base_frame = frames[0]
    rgba_frame = np.dstack((base_frame, np.full((height, width), 255, dtype=np.uint8)))
    flat_rgba = rgba_frame.reshape(-1, 4)
    rgba_pixels = [tuple(pixel) for pixel in flat_rgba]

    # Use polygon mode for blocky Atari graphics and disable filters for max detail.
    svg_string = vtracer.convert_pixels_to_svg(
        rgba_pixels, (width, height), mode="none", filter_speckle=0, length_threshold=0.0
    )

    match = re.search(r"<svg[^>]*>(.*)</svg>", svg_string, re.DOTALL)
    if match:
        # The base frame is always visible.
        svg_parts.append(f'<g visibility="visible">{match.group(1)}</g>')

    # --- Vectorize and add differential frames ---
    print("Vectorizing differential frames...")
    prev_frame = base_frame
    for i, curr_frame in tqdm(enumerate(frames[1:], start=1), total=len(frames) - 1, desc="Processing Diffs"):
        # Find differences between current and previous frame.
        diff_mask = (prev_frame != curr_frame).any(axis=2)

        # If there's no change, we don't need to add a layer.
        if not np.any(diff_mask):
            prev_frame = curr_frame
            continue

        # Create a transparent frame, copying only the changed pixels.
        diff_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        diff_rgba[diff_mask, :3] = curr_frame[diff_mask]  # Copy RGB from current frame.
        diff_rgba[diff_mask, 3] = 255  # Set Alpha to opaque for changed pixels.

        # Vectorize the sparse differential frame.
        flat_diff_rgba = diff_rgba.reshape(-1, 4)
        diff_pixels = [tuple(pixel) for pixel in flat_diff_rgba]
        diff_svg_string = vtracer.convert_pixels_to_svg(
            diff_pixels, (width, height), mode="none", filter_speckle=0, length_threshold=0.0
        )

        match = re.search(r"<svg[^>]*>(.*)</svg>", diff_svg_string, re.DOTALL)
        if match:
            svg_content = match.group(1)
            if svg_content.strip():  # Only add a group if there's vector content.
                begin_time = i * delay_s
                svg_parts.append('<g visibility="hidden">')
                svg_parts.append(f'  <set attributeName="visibility" to="visible" begin="{begin_time:.4f}s" />')
                svg_parts.append(svg_content)
                svg_parts.append("</g>")

        prev_frame = curr_frame

    svg_parts.append("</g>")  # Close the main container group.
    svg_parts.append("</svg>")  # Close the svg.

    print("Writing SVG file...")
    with video_path.open("w", encoding="utf-8") as f:
        f.write("".join(svg_parts))

    print(f"Differentially rendered SVG animation saved successfully to {video_path}")


def _save_to_mp4(frames: list[np.ndarray], video_path: Path):
    """Saves a list of frames as a high-quality MP4 video using OpenCV."""
    print(f"Saving to MP4 using OpenCV: {video_path}")
    if not frames:
        print("Warning: No frames to save for MP4.", file=sys.stderr)
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
    print(f"MP4 video saved successfully to {video_path}")


def create_video(exp_id: int | None, video_format: str):
    """Constructs and runs the video recording command for the specified format."""
    env_path_name = config.ENV.replace("/", "-")

    # rl-zoo3's get_model_path expects exp_id=0 to find the latest experiment,
    # but receives None from argparse if the flag is not set.
    effective_exp_id = exp_id if exp_id is not None else 0

    # The get_model_path function returns a tuple of (name_prefix, model_path, log_path).
    # We use the log_path to ensure videos are saved in the correct experiment directory.
    _name_prefix, model_path_str, log_path_str = get_model_path(
        exp_id=effective_exp_id,
        folder=config.LOG_FOLDER,
        algo=config.ALGO,
        env_name=env_path_name,
        load_best=True,
    )
    model_path = Path(model_path_str)
    log_path = Path(log_path_str)

    print(f"Loading model from: {model_path}")

    # Generate high-quality frames. This is now the single source of truth for all formats.
    frames = _generate_hq_frames(model_path)
    if not frames:
        print("Error: No frames were recorded.", file=sys.stderr)
        sys.exit(1)

    # The original rl_zoo3 script saves videos to a 'videos' subfolder within the experiment's log directory.
    video_folder = log_path / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)
    base_filename = f"replay_{config.ALGO}_{env_path_name}_steps_{config.RECORD_TIMESTEPS}"

    if video_format in ["svg", "all"]:
        svg_path = video_folder / f"{base_filename}.svg"
        _save_to_svg(frames, svg_path)

    if video_format in ["mp4", "all"]:
        mp4_path = video_folder / f"{base_filename}.mp4"
        _save_to_mp4(frames, mp4_path)
