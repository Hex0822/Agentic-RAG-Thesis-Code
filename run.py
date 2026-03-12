import logging
import os
import sys
from datetime import datetime


def _ensure_project_root_on_path() -> None:
    """Allow running `python process/run.py` from the repo root."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _configure_logging(level: int, log_path: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _shutdown_logging() -> None:
    logger = logging.getLogger()
    handlers = list(logger.handlers)
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

def main() -> None:
    _ensure_project_root_on_path()
    from process.config import get_log_level
    from process.pipeline import run_demo

    logs_dir = _ensure_logs_dir()
    temp_log_path = os.path.join(logs_dir, "process_running.log")
    log_level = getattr(logging, get_log_level(), logging.INFO)
    _configure_logging(log_level, temp_log_path)

    logger = logging.getLogger(__name__)
    logger.info("Starting process pipeline...")
    try:
        run_demo()
    finally:
        logger.info("Process pipeline finished.")
        _shutdown_logging()
        end_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_path = os.path.join(logs_dir, f"process_{end_stamp}.log")
        try:
            os.replace(temp_log_path, final_log_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
