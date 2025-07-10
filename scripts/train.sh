#!/bin/sh
# Source the shared configuration
. "$(dirname "$0")/config.sh"

RED='\033[0;31m'
NC='\033[0m' # No Color

if [ "$1" = "--resume" ]; then
    # --- Resume Training Logic ---
    get_latest_model() {
        # Construct the base path name for the environment (e.g., "ALE/Breakout-v5" -> "ALE-Breakout-v5")
        local ENV_PATH_NAME
        ENV_PATH_NAME=$(echo "$ENV" | tr '/' '-')

        # Find the latest experiment directory for the current algorithm and environment.
        # It sorts by version number to correctly handle exp_id > 9.
        local LATEST_EXP_DIR
        LATEST_EXP_DIR=$(ls -d -1 "$LOG_FOLDER/$ALGO/${ENV_PATH_NAME}_"* 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LATEST_EXP_DIR" ]; then
            echo "${RED}get_latest_model error:${NC} No experiment folder found for “$ALGO” on “$ENV” in “$LOG_FOLDER”." >&2
            echo "- Please run a training session first with './scripts/train.sh'.\n" >&2
            return 1
        fi

        # The model saved at the end of training is named after the environment.
        local FINAL_MODEL_PATH="$LATEST_EXP_DIR/$ENV_PATH_NAME.zip"
        local BEST_MODEL_PATH="$LATEST_EXP_DIR/best_model.zip"

        # Check if the final model file exists. If not, fall back to the best model.
        if [ -f "$FINAL_MODEL_PATH" ]; then
            echo "$FINAL_MODEL_PATH"
        elif [ -f "$BEST_MODEL_PATH" ]; then
            echo "$BEST_MODEL_PATH"
        else
            echo "${RED}Get_latest_model Error:${NC} No model file found in “$LATEST_EXP_DIR” to resume training." >&2
            return 1
        fi
    }

    TRAINED_AGENT=$(get_latest_model)
    if [ $? -eq 0 ]; then
        echo "Resuming training from: $TRAINED_AGENT"
        run_training --trained-agent "$TRAINED_AGENT"
    else
        echo "${RED}Train Error:${NC} Failed to find a trained agent to resume from." >&2
        exit 1
    fi
else
    # --- Start New Training ---
    run_training
fi