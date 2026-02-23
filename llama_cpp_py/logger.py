import os
import sys
from loguru import logger


class IsolationDefaultHandlerFilter:
    """
    A filter for isolating the default logging handler (which is initialized automatically)
    to avoid having to call logger.remove() (as is commonly done and recommended in the 
    documentation) while preventing the default handler from writing unnecessary log entries.

    Args:
        logger_extras: A list of extra values for loggers that should 
                       not be logged by the default auto-initialized handler
    """
    def __init__(self, logger_extras: list[str]):
        self.logger_extras = logger_extras

    def __call__(self, record):
        for logger_extra in self.logger_extras:
            if logger_extra in record['extra']:
                return False
        return True


if 0 in logger._core.handlers:
    logger._core.handlers[0]._filter = IsolationDefaultHandlerFilter(
        logger_extras=['llama_server', 'llama_debug']
    )


server_logger = logger.bind(llama_server=True)
debug_logger = logger.bind(llama_debug=True)

log_level = os.getenv('LLAMACPP_LOG_LEVEL', 'INFO')

server_logger.add(
    sink=sys.stderr,
    level=log_level,
    filter=lambda record: 'llama_server' in record['extra'],
)

debug_logger.add(
    sink=sys.stderr,
    level=log_level,
    filter=lambda record: 'llama_debug' in record['extra'],
)
