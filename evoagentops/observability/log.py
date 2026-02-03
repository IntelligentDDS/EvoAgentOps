import os
import logging
from datetime import datetime
from pathlib import Path
from loguru import logger


def agent_logger(level: str = "INFO", log_dir: str = "./logs"):
    os.environ["TZ"] = "UTC"
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    # Output to agent-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{timestamp}.log"

    """Configure multi-agent system logger - using loguru"""
    global logger
    # Remove default handler
    logger.remove()
    # Unified format
    loguru_log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] (Process-{process.id} {name}) {message}"
    # Output to console
    logger.add(sink=lambda msg: print(msg, end=""), level=level, format=loguru_log_format, colorize=True)
    logger.add(
        sink=str(log_file),
        level=level,
        format=loguru_log_format,
        rotation="00:00",  # Rotate daily
        retention="7 days",  # Keep for 7 days
        encoding="utf-8",
    )

    """Configure multi-agent system logger - using logging"""
    global logging
    logging_logger = logging.getLogger()
    logging_logger.handlers.clear()  # Clear existing handlers
    # Unified format
    logging_formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s) %(message)s")
    # Output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging_formatter)
    logging_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging_formatter)
    logging_logger.addHandler(file_handler)

    logging_logger.setLevel(level)
    logging_logger.propagate = False  # Prevent duplicate logs


# agent_logger()
