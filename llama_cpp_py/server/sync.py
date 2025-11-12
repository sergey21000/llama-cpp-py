import os
import io
import time
import threading
import subprocess
import platform
import logging
import requests
from pathlib import Path

from llama_cpp_py.logger import logger, status_logger
from llama_cpp_py.release_manager.manager import LlamaReleaseManager
from llama_cpp_py.server.base import LlamaBaseServer


class LlamaSyncServer(LlamaBaseServer):
    """Synchronous implementation of llama.cpp server manager.
    
    Manages server process using subprocess.Popen with threaded output logging.
    Suitable for synchronous applications and scripts.
    """
    def __init__(
        self,
        llama_dir: str | Path = '',
        release_manager: LlamaReleaseManager | None = None,
        verbose: bool = False,
        wait_for_ready: bool = True,
        **subprocess_kwargs,
    ):
        """Initialize llama server instance.
        
        Args:
            llama_dir: Directory containing llama-server executable
            host: Server host address (default: 127.0.0.1)
            port: Server port (default: 8080)
            release_manager: Optional pre-configured release manager
            verbose: Enable verbose logging of server output
            wait_for_ready: If True, waits for server health check before start() completes;
                           if False, returns immediately after process launch
            **subprocess_kwargs: Additional arguments for subprocess.Popen 
                https://docs.python.org/3/library/subprocess.html#popen-constructor
        """
        super().__init__(
            llama_dir=llama_dir,
            release_manager=release_manager,
            verbose=verbose,
            wait_for_ready=wait_for_ready,
            **subprocess_kwargs,
        )


    def start(self) -> None:
        """Start the llama.cpp server synchronously."""
        status_logger.info('llama.cpp server starting ...')
        self.process = subprocess.Popen(
            [self.start_server_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **self.subprocess_kwargs,
        )
        if self.verbose:
            threading.Thread(
                target=self.log_output,
                kwargs=dict(stream=self.process.stdout),
                daemon=True,
            ).start()
            threading.Thread(
                target=self.log_output,
                kwargs=dict(stream=self.process.stderr),
                daemon=True,
            ).start()
        if not self.wait_for_ready:
            status_logger.info('Server process started (not waiting for readiness)')
            return
        try:
            server_is_ready = self.wait_for_server_ready(
                url=self.health_url,
                timeout=self.timeout_wait_for_server_ready,
            )
            if not server_is_ready:
                self.stop()
                raise TimeoutError('Server did not start within the allotted time')
            status_logger.info('llama.cpp server ready')
        except Exception:
            self.stop()
            raise


    def stop(self) -> None:
        """Stop the llama.cpp server gracefully with fallback to force kill."""
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=self.timeout_to_stop_process)
            logger.info('llama.cpp server stopped correctly')
        except subprocess.TimeoutExpired:
            logger.info('The server did not respond to terminate(), killing it ...')
            self.process.kill()
            self.process.wait()
        except ProcessLookupError:
            logger.info('Process already terminated, nothing to stop')
        self.process = None
        status_logger.info('llama.cpp server stopped')


    def log_output(self, stream: io.BufferedReader, log_prefix: str = '') -> None:
        """Log server output from the given stream in a separate thread.
    
        Handles both regular output lines and dynamic progress updates. Progress lines
        (ending with carriage return) are updated in-place, while regular lines
        are printed on new lines.
        
        Args:
            stream: Binary stream to read output from (stdout/stderr)
            log_prefix: Optional prefix to add to each output line for identification
        """
        state = dict(buffer=b'', last_was_cr=False)
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            self.process_log_output_chunk(chunk, state, log_prefix)
        if state['last_was_cr']:
            print()
        stream.close()


    def wait_for_server_ready(self, url: str, timeout: int | float = 60) -> bool:
        """Wait synchronously for server to become ready and respond to health checks."""
        start_time = time.monotonic()
        model_loading = False
        while time.monotonic() - start_time < timeout:
            ret = self.process.poll() 
            if ret is not None:
                status_logger.error(f'llama.cpp process exited unexpectedly with code {ret}')
                return False
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
                elif response.status_code == 503:
                    if not model_loading:
                        model_loading = True
                        status_logger.info('Model is loading (503), waiting...')
                    start_time = time.monotonic()
                else:
                   logger.debug(f'Unexpected status code {response.status_code}, retrying...')
            except requests.RequestException as e:
                logger.debug(f'Connection error: {e}, retrying...')
            time.sleep(1)
        status_logger.warning(f'Server did not become ready within {timeout}s')
        return False


    def __enter__(self):
        """Start the server when entering a context manager."""
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the server when exiting a context manager."""
        self.stop()
