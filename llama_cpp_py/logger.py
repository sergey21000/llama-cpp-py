import os
import sys
from loguru import logger


if 0 in logger._core.handlers:
    logger.remove(0)
    logger.add(
        sink=sys.stderr,
        level='INFO',
        filter=lambda record: record['extra'].get('name') == 'llama_server',
    )

server_logger = logger.bind(name=llama_server)
debug_logger = logger.bind(name=llama_debug)

log_level = os.getenv('LLAMACPP_LOG_LEVEL')
if log_level:
    debug_logger.add(
        sink=sys.stderr,
        level=log_level,
        filter=lambda record: record['extra'].get('name') == 'llama_debug',
    )
