import os
import sys
from loguru import logger


if 0 in logger._core.handlers:
    logger.remove(0)
    logger.add(
        sink=sys.stderr,
        level='INFO',
        filter=lambda record: 'llama_server' in record['extra'],
    )

server_logger = logger.bind(llama_server=True)
debug_logger = logger.bind(llama_debug=True)

log_level = os.getenv('LLAMACPP_LOG_LEVEL')
if log_level:
    debug_logger.add(
        sink=sys.stderr,
        level=log_level,
        filter=lambda record: 'llama_debug' in record['extra'],
    )
