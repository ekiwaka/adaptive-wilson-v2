from __future__ import annotations
import logging
from pathlib import Path
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'


def _find_file_handler(logger: logging.Logger, log_file: Path) -> logging.FileHandler | None:
    target = str(log_file.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == Path(target):
                    return handler
            except Exception:
                continue
    return None


def _has_stdout_handler(logger: logging.Logger) -> bool:
    return any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers)


def configure_logging(
    level: str,
    log_file: str | Path,
    *,
    log_to_stdout: bool = True,
    reset_handlers: bool = False,
) -> logging.Logger:
    """Configure root logging for one experiment process.

    When reset_handlers is false, existing handlers are preserved so study-level
    logging can coexist with per-run file logging.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    if reset_handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if log_to_stdout and not _has_stdout_handler(root_logger):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(stream_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _find_file_handler(root_logger, log_path) is None:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)

    return root_logger


def add_file_logging_handler(level: str, log_file: str | Path) -> logging.Handler:
    """Attach one additional file handler to the root logger and return it."""

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _find_file_handler(root_logger, log_path)
    if existing is not None:
        return existing

    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setLevel(level.upper())
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(handler)
    return handler


def remove_logging_handler(handler: logging.Handler | None) -> None:
    """Detach and close a handler if it is currently registered."""

    if handler is None:
        return
    root_logger = logging.getLogger()
    if handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)
