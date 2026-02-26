import os
import io
import sys
import asyncio
import time
import threading
import subprocess
import platform
from pathlib import Path

if platform.system() != 'Windows':
    import pty

import aiohttp

from llama_cpp_py.logger import debug_logger, server_logger
from llama_cpp_py.release_manager.manager import LlamaReleaseManager
from llama_cpp_py.server.base import LlamaBaseServer


class LlamaAsyncServer(LlamaBaseServer):
    """
    Asynchronous implementation of llama.cpp server manager.

        Args:
            llama_dir: Directory containing llama-server executable
                (env: LLAMACPP_DIR)
            host: Server host address (default: 127.0.0.1)
            port: Server port (default: 8080)
            release_manager: Optional pre-configured release manager
            verbose: Enable verbose logging of server output
            wait_for_ready: If True, waits for server health check before start() completes;
                           if False, returns immediately after process launch
            **subprocess_kwargs: Additional arguments for subprocess.Popen 
                https://docs.python.org/3/library/subprocess.html#popen-constructor
    """

    def __init__(
        self,
        llama_dir: str | Path = '',
        release_manager: LlamaReleaseManager | None = None,
        verbose: bool = True,
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
            **subprocess_kwargs: Additional arguments for 
                asyncio.create_subprocess_exec
                https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.subprocess_exec
        """
        super().__init__(
            llama_dir=llama_dir,
            release_manager=release_manager,
            verbose=verbose,
            wait_for_ready=wait_for_ready,
            **subprocess_kwargs,
        )

    async def start(self) -> bool:
        """Start the llama.cpp server asynchronously."""
        server_logger.info('llama.cpp server starting ...')
        if not self.verbose:
            self.process = await asyncio.create_subprocess_exec(
                self.start_server_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                **self.subprocess_kwargs,
            )
        else:
            if not self.is_jupyter_runtime():
                self.process = await asyncio.create_subprocess_exec(
                    self.start_server_cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    **self.subprocess_kwargs,
                )
            else:
                master_fd, slave_fd = pty.openpty()
                self.process = await asyncio.create_subprocess_exec(
                    self.start_server_cmd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    **self.subprocess_kwargs,
                )
                os.close(slave_fd)
                self.pty_master_fd = master_fd
                threading.Thread(
                    target=self.log_output_pty,
                    daemon=True,
                ).start()
        if not self.wait_for_ready:
            server_logger.info('Server process started (not waiting for readiness)')
            return
        try:
            server_is_ready = await self.wait_for_server_ready(
                url=self.health_url,
                timeout=self.timeout_wait_for_server_ready,
            )
            if not server_is_ready:
                await self.stop()
                raise TimeoutError('Server did not start within the allotted time')
            server_logger.info('llama.cpp server ready')
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> bool:
        """Stop the llama.cpp server asynchronously with proper cleanup."""
        if not self.process:
            return
        try:
            self.process.terminate()
            await asyncio.wait_for(
                self.process.wait(),
                timeout=self.timeout_to_stop_process,
            )
            debug_logger.info('llama.cpp server stopped correctly')
        except asyncio.TimeoutError:
            debug_logger.info('The server did not respond to terminate(), killing it ...')
            self.process.kill()
            await self.process.wait()
        except ProcessLookupError:
            debug_logger.info('Process already terminated, nothing to stop')
        self.process = None
        server_logger.info('llama.cpp server stopped')

    async def wait_for_server_ready(self, url: str, timeout: int | float) -> bool:
        """Wait asynchronously for server to become ready and respond to health checks."""
        async with aiohttp.ClientSession() as session:
            start_time = time.monotonic()
            model_loading = False
            while time.monotonic() - start_time < timeout:
                ret = self.process.returncode
                if ret is not None:
                    server_logger.error(f'llama.cpp process exited unexpectedly with code {ret}')
                    return False
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            return True
                        elif response.status == 503:
                            if not model_loading:
                                model_loading = True
                                server_logger.info('Model is loading (503), waiting...')
                            start_time = time.monotonic()
                        else:
                            debug_logger.debug(f'Unexpected status code {response.status}, retrying...')
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    debug_logger.debug(f'Connection error: {e}, retrying...')
                await asyncio.sleep(1)
        server_logger.warning(f'Server did not become ready within {timeout}s')
        return False

    async def __aenter__(self):
        """Start the server when entering an async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the server when exiting an async context manager."""
        await self.stop()
