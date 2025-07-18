from pathlib import Path

from atari_rl import config
from atari_rl.logger import logger


def find_experiment_folder(exp_id: int | None) -> Path | None:
    """Finds an experiment folder, or the latest one if no ID is provided.

    Args:
        exp_id: The specific experiment ID to find. If None, finds the latest.

    Returns:
        The path to the experiment folder, or None if not found.

    """
    env_path_name = config.ENV.replace("/", "-")
    log_path = Path(config.LOG_FOLDER)
    algo_log_path = log_path / config.ALGO

    if not algo_log_path.is_dir():
        logger.error(f"Log directory for algorithm '{config.ALGO}' not found at: {algo_log_path}")
        return None

    if exp_id is not None:
        # Find the specific experiment folder by ID.
        exp_folders = list(algo_log_path.glob(f"{env_path_name}_{exp_id}*"))
        if not exp_folders:
            logger.error(f"Experiment folder for ID {exp_id} not found in {algo_log_path}.")
            return None
        return exp_folders[0]

    # If no exp_id, find the latest one.
    exp_dirs = list(algo_log_path.glob(f"{env_path_name}_*"))
    if not exp_dirs:
        logger.error(f"No experiment folder found for '{config.ALGO}' on '{config.ENV}' in '{algo_log_path}'.")
        logger.error("Please run a training session first with 'atari-rl train'.")
        return None

    # Sort directories by experiment number to find the latest
    latest_exp_dir = max(exp_dirs, key=lambda p: int(p.name.split("_")[-1]))
    return latest_exp_dir


def find_model_path(experiment_folder: Path, prefer_best: bool = True) -> Path | None:
    """Finds a model file within an experiment folder with fallback logic.

    Args:
        experiment_folder: The path to the experiment folder.
        prefer_best: If True, prioritizes 'best_model.zip'. If False,
                     prioritizes the final model for resuming training.

    Returns:
        The path to the model file, or None if not found.

    """
    env_path_name = config.ENV.replace("/", "-")
    best_model = experiment_folder / "best_model.zip"
    final_model = experiment_folder / f"{env_path_name}.zip"

    primary_model = best_model if prefer_best else final_model
    fallback_model = final_model if prefer_best else best_model

    if primary_model.is_file():
        return primary_model
    elif fallback_model.is_file():
        logger.info(f"Could not find {primary_model.name}, falling back to {fallback_model.name}.")
        return fallback_model
    else:
        logger.error(f"No model file ('{primary_model.name}' or '{fallback_model.name}') found in '{experiment_folder}'.")
        return None


def find_model_for_evaluation(exp_id: int | None) -> tuple[Path, Path] | tuple[None, None]:
    """Finds the best model for evaluation (enjoy/record-video).

    - If exp_id is given, it looks for the model in that specific experiment,
      with a fallback to the final model if 'best_model.zip' is not present.
    - If exp_id is None, it searches all experiments for the latest 'best_model.zip'.
    - If no 'best_model.zip' is found anywhere, it falls back to the latest model
      (best or final) in the most recent experiment.

    Returns:
        A tuple of (model_path, experiment_folder_path), or (None, None).

    """
    # --- Case 1: A specific experiment ID is provided ---
    if exp_id is not None:
        exp_folder = find_experiment_folder(exp_id)
        if not exp_folder:
            return None, None  # find_experiment_folder already logs error
        model_path = find_model_path(exp_folder, prefer_best=True)
        if not model_path:
            return None, None  # find_model_path already logs error
        return model_path, exp_folder

    # --- Case 2: No experiment ID, find the absolute best model across all runs ---
    env_path_name = config.ENV.replace("/", "-")
    log_path = Path(config.LOG_FOLDER)
    algo_log_path = log_path / config.ALGO

    if not algo_log_path.is_dir():
        logger.error(f"Log directory for algorithm '{config.ALGO}' not found at: {algo_log_path}")
        return None, None

    # Get all experiment directories, sorted from latest to oldest
    all_exp_dirs = sorted(algo_log_path.glob(f"{env_path_name}_*"), key=lambda p: int(p.name.split("_")[-1]), reverse=True)

    if not all_exp_dirs:
        logger.error(f"No experiment folder found for '{config.ALGO}' on '{config.ENV}' in '{algo_log_path}'.")
        return None, None

    logger.info("No --exp-id provided. Searching for the best model across all experiments...")
    for exp_folder in all_exp_dirs:
        best_model_path = exp_folder / "best_model.zip"
        if best_model_path.is_file():
            logger.info(f"Found best model in experiment: {exp_folder.name}")
            return best_model_path, exp_folder

    # --- Fallback: No 'best_model.zip' found anywhere ---
    logger.warning("No 'best_model.zip' found in any experiment.")
    latest_exp_folder = all_exp_dirs[0]
    logger.info(f"Falling back to the latest model in the most recent experiment: {latest_exp_folder.name}")
    model_path = find_model_path(latest_exp_folder, prefer_best=True)
    if model_path:
        return model_path, latest_exp_folder

    logger.error(f"Could not find any model in the latest experiment folder: {latest_exp_folder}")
    return None, None
