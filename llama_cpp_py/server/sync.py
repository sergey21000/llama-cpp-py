import os
import io
import time
import threading
import subprocess
import platform
import logging
import requests
from pathlib import Path

from llama_cpp_py.logger import logger
from llama_cpp_py.release_manager.manager import LlamaReleaseManager
from llama_cpp_py.server.base import LlamaBaseServer


class LlamaSyncServer(LlamaBaseServer):
    def __init__(
        self,
        llama_dir: str | Path = '',
        host: str = '',
        port: int | str = '',
        release_manager: LlamaReleaseManager | None = None,
        verbose: bool = True,
    ):
        super().__init__(
            llama_dir=llama_dir,
            host=host,
            port=port,
            release_manager=release_manager,
            verbose=verbose,
        )

    def start(self, **subprocess_popen_kwargs):
        logger.info('llama.cpp server starting  ...')
        self.process = subprocess.Popen(
            [self.start_server_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_popen_kwargs,
        )
        if self.verbose:
            self.threads = [
                threading.Thread(
                    target=self.log_output,
                    kwargs=dict(stream=self.process.stdout),
                    daemon=True,
               ),
                threading.Thread(
                    target=self.log_output,
                    kwargs=dict(stream=self.process.stderr),
                    daemon=True,
               ),
            ]
            for thread in self.threads:
                thread.start()
        self.wait_for_server_ready(self.health_url)

    def stop(self):
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.timeout_to_stop_process)
            logger.info('llama.cpp server stopped')
        except subprocess.TimeoutExpired:
            logger.info('The server did not respond to terminate(), killing it ...')
            self.process.kill()
            self.process.wait()

    def log_output(self, stream: io.BufferedReader):
        for line in iter(stream.readline, b''):
            try:
                decoded_line = line.decode().rstrip()
                self.process_logger.info(decoded_line)
            except UnicodeDecodeError:
                pass
        stream.close()

    @staticmethod
    def wait_for_server_ready(url: str, timeout: float = 60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logger.info('Server is ready')
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError('Server did not start within the allotted time')