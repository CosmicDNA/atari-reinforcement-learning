import logging
import sys

# ANSI color codes
GRAY = "\033[0;90m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
NC = "\033[0m"  # No Color


class ColorFormatter(logging.Formatter):
    """A logging formatter that adds color to log levels."""

    # The message format for each level.
    FORMATS = {
        logging.DEBUG: f"{GRAY}%(message)s{NC}",
        logging.INFO: "%(message)s",
        logging.WARNING: f"{YELLOW}Warning: %(message)s{NC}",
        logging.ERROR: f"{RED}Error: %(message)s{NC}",
        logging.CRITICAL: f"{RED}Critical: %(message)s{NC}",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Applies the correct color format to the log record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger() -> logging.Logger:
    """Sets up a colored logger that separates INFO and WARNING/ERROR streams."""
    logger = logging.getLogger("atari_rl")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if this function is called repeatedly.
    if logger.hasHandlers():
        return logger

    # Handler for stdout (for INFO/DEBUG) and stderr (for WARNING/ERROR)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    stdout_handler.setFormatter(ColorFormatter())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(ColorFormatter())

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


# Create a single logger instance to be used across the application.
logger = setup_logger()
