# Centralized logging for dispatch module.

import logging
import os

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_loggers: dict = {}


def get_logger(name: str = "sglang_fl.dispatch") -> logging.Logger:
    """Get or create a logger for the dispatch module."""
    if name not in _loggers:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        level_str = os.environ.get("SGLANG_FL_LOG_LEVEL", "INFO").upper()
        logger.setLevel(_LOG_LEVEL_MAP.get(level_str, logging.INFO))
        _loggers[name] = logger

    return _loggers[name]


def set_log_level(level: str, name: str = "sglang_fl.dispatch") -> None:
    """Set log level for a dispatch logger."""
    logger = get_logger(name)
    logger.setLevel(_LOG_LEVEL_MAP.get(level.upper(), logging.INFO))
