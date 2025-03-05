import logging
import sys
from typing import Optional

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def create_logger(
    name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    if name is None:
        raise ValueError("name for logger cannot be None")

    formatter = logging.Formatter(
        "[%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    logger_.propagate = False
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger_.addHandler(ch)
    return logger_
