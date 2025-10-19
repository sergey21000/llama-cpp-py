import os
import platform
import logging
from pathlib import Path

from llama_cpp_py.release_manager.manager import LlamaReleaseManager


class LlamaBaseServer:
    def __init__(
        self,
        llama_dir: str | Path = '',
        host: str = '',
        port: int | str = '',
        release_manager: LlamaReleaseManager | None = None,
        verbose: bool = True,
    ):
        self.host = host or os.getenv('LLAMA_ARG_HOST') or 'localhost'
        self.port = port or os.getenv('LLAMA_ARG_PORT') or '8080'
        os.environ['LLAMA_ARG_HOST'] = self.host
        os.environ['LLAMA_ARG_PORT'] = self.port
        self.server_url = f'http://{self.host}:{self.port}'
        self.health_url  = f'{self.server_url}/health'
        self.timeout_to_stop_process = 3
        if not llama_dir:
            if not release_manager:
                release_manager = LlamaReleaseManager()
            llama_dir = release_manager.release_dir
        self.llama_dir = Path(llama_dir)
        lib_file = 'llama-server.exe' if platform.system() == 'Windows' else 'llama-server'
        self.start_server_cmd = str(self.llama_dir / lib_file)
        self.process = None
        self.verbose = verbose
        if self.verbose:
            self.process_logger = self._get_process_logger()


    def _get_process_logger(self):
        logger = logging.getLogger('llama-server')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
