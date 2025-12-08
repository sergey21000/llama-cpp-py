import sys
import logging


def get_base_logger(name: str, debug: bool) -> logging.Logger:
    """Get base logger with debug mode support.
    
    Args:
        name: Logger name
        debug: If True, creates detailed logger with file/line info;
               if False, returns logger with NullHandler
    
    Returns:
        Configured logger instance
    """
    if not debug:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger.addHandler(logging.NullHandler())
        return logger
    format = (
        '%(name)s: %(asctime)s: %(levelname)s: '
        '%(filename)s:%(lineno)d: %(funcName)s: %(message)s'
    )
    return get_stream_logger(
        name=name,
        format=format,
        level=logging.DEBUG,
    )


def get_stream_logger(name: str, format: str, level: int) -> logging.Logger:
    """Create and configure a stream logger with specific format and level.
    
    Args:
        name: Logger name
        format: Log message format string
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    
    Returns:
        Configured logger instance with StreamHandler
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# main package logger for general debugging and internal operations
logger = get_base_logger(name='llama-cpp-py', debug=False)

# logger for llama-server process output and system messages
process_logger = get_stream_logger(
    name='llama-server',
    format='%(name)s: %(message)s',
    level=logging.INFO,
)

# logger for server status messages (startup, shutdown, health checks)
status_logger = get_stream_logger(
    name='llama-cpp-py.status',
    format='%(levelname)s: %(message)s',
    level=logging.INFO,
)
