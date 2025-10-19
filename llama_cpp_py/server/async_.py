import os
import io
import asyncio
import threading
import subprocess
import platform
from pathlib import Path

import aiohttp

from llama_cpp_py.logger import logger
from llama_cpp_py.release_manager.manager import LlamaReleaseManager
from llama_cpp_py.server.base import LlamaBaseServer


class LlamaAsyncServer(LlamaBaseServer):
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

    async def start(self, **subprocess_exec_kwargs):
        self.process = await asyncio.create_subprocess_exec(
            *self.start_server_cmd.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **subprocess_exec_kwargs,
        )
        logger.info('llama.cpp server starting ...')
        if self.verbose:
            asyncio.create_task(self.log_output(stream=self.process.stdout))
            asyncio.create_task(self.log_output(stream=self.process.stderr))
        await self.wait_for_server_ready(url=self.health_url)

    async def stop(self):
        if not self.process:
            return
        try:
            self.process.terminate()
            await asyncio.wait_for(
                self.process.wait(),
                timeout=self.timeout_to_stop_process,
            )
            logger.info('llama.cpp server stopped')
        except asyncio.TimeoutError:
            logger.info('The server did not respond to terminate(), killing it ...')
            self.process.kill()
            await self.process.wait()
        self.process = None

    async def log_output(self, stream: io.BufferedReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                decoded_line = line.decode().rstrip()
                self.process_logger.info(decoded_line)
            except UnicodeDecodeError:
                pass

    @staticmethod
    async def wait_for_server_ready(url: str, timeout: float = 60):
        async with aiohttp.ClientSession() as session:
            for _ in range(timeout):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            logger.info('Server is ready')
                            return True
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError('Server did not start within the allotted time')
