"""Information generating functions.

Contain functions used to generate information in the script: loggers,
tags...
"""

from pathlib import Path
from typing import Any

import subprocess
import logging
import sys

from datetime import datetime


def config_logger(log_dir: Path | None = None) -> logging.Logger:
    """Logging function. It has two main handlers:

        - a StreamHandler that logs INFO level log in stdout
        - an optional FileHandler that logs DEBUG level log in a .log file

    Args:
        log_dir (Path): output directory where to store log.
        If not passed, logs will only be printed in stdout

    Returns:
        logger: main logger
    """
    logger = logging.getLogger(name="standard-logger")
    # adding handler to stdout
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # DEBUG as default
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s - [%(process)s] %(message)s"
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)  # stdout prints only INFO
        # adding handler to .log file
        if log_dir is not None:
            # add timestamp to log path
            timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"logs_{timestamp}.log"

            # set .log path
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)

            # log contains also debug
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    return logger


def get_tags() -> dict[str, Any]:
    """Generate dictionary of mlflow tags."""
    tags: dict[str, Any] = {}

    # retrieve git branch
    git_branch: str = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tags["git_branch"] = git_branch

    # retrieve git commit hash
    git_commit: str = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tags["git_commit"] = git_commit

    # get username - using git one
    user: str = subprocess.run(
        ["git", "config", "user.name"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tags["user"] = user

    # get environment
    tags["environment"] = "test"

    return tags
