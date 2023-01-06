"""Logging.

This module contains logging functions
"""

from pathlib import Path

import logging
import sys

from datetime import datetime


def config_logger(log_dir: Path | None = None) -> logging.RootLogger:
    """Logging function. It has two main handlers:

        - a StreamHandler that logs INFO level log in stdout
        - an optional FileHandler that logs DEBUG level log in a .log file

    Args:
        log_dir (Path): output directory where to store log.
        If not passed, logs will only be printed in stdout

    Returns:
        logger: main logger
    """
    logger = logging.getLogger()
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
