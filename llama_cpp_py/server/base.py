import os
import io
import sys
import platform
from pathlib import Path

from llama_cpp_py.logger import debug_logger
from llama_cpp_py.release_manager.manager import LlamaReleaseManager


class LlamaBaseServer:
    """Base server class for managing llama.cpp server instances.
    
    Handles common configuration, process management, and health checking
    for both synchronous and asynchronous server implementations.
    """
    def __init__(
        self,
        llama_dir: str | Path = '',
        release_manager: LlamaReleaseManager | None = None,
        verbose: bool = False,
        wait_for_ready: bool = True,
        **subprocess_kwargs,
    ):
        """Initialize the base llama.cpp server instance.
        
        Args:
            llama_dir: Directory containing llama-server executable. If not provided,
                     uses the release manager's directory.
                (env: LLAMACPP_DIR)
            release_manager: Pre-configured LlamaReleaseManager instance for 
                           automatic binary management. Created if not provided.
            verbose: Enable verbose logging of server output to stdout/stderr
            wait_for_ready: Whether to wait for server health check before returning 
                          from start methods
            **subprocess_kwargs: Additional arguments passed to subprocess.Popen/
                               create_subprocess_exec. Use 'env' key to provide
                               custom environment variables.
        
        Raises:
            ValueError: If LLAMA_ARG_HOST or LLAMA_ARG_PORT environment variables 
                       are not set in either os.environ or subprocess_kwargs['env']
        
        Note:
            Environment variables are read from subprocess_kwargs['env'] if provided,
            otherwise from os.environ. The following variables are required:
            - LLAMA_ARG_HOST: Server host address (e.g., '127.0.0.1')
            - LLAMA_ARG_PORT: Server port (e.g., '8080')
        """
        subprocess_kwargs.setdefault('env', os.environ)
        env = subprocess_kwargs['env']
        self.host = env.get('LLAMA_ARG_HOST')
        self.port = env.get('LLAMA_ARG_PORT')
        if not self.host or not self.port:
            raise ValueError(
                'LLAMA_ARG_HOST and LLAMA_ARG_PORT environment variables must be set. '
                f'Got: LLAMA_ARG_HOST={repr(self.host)}, LLAMA_ARG_PORT={repr(self.port)}\n'
                'Set them in:\n'
                '1. os.environ\n'
                "2. env parameter: LlamaSyncServer(env={'LLAMA_ARG_HOST': '...', 'LLAMA_ARG_PORT': '...'})\n"
                '3. .env file with load_dotenv()'
            )
        self.server_url = f'http://{self.host}:{self.port}'
        self.openai_base_url = f'{self.server_url}/v1'
        self.health_url  = f'{self.server_url}/health'
        timeout_wait_for_server_ready = os.getenv('LLAMACPP_SERVER_TIMEOUT_WAIT') or 300
        self.timeout_wait_for_server_ready = int(timeout_wait_for_server_ready)
        self.timeout_to_stop_process = 3
        llama_dir = llama_dir or os.getenv('LLAMACPP_DIR')
        if not llama_dir:
            if not release_manager:
                release_manager = LlamaReleaseManager()
            llama_dir = release_manager.release_dir
        self.llama_dir = Path(llama_dir)
        lib_file = 'llama-server.exe' if platform.system() == 'Windows' else 'llama-server'
        self.start_server_cmd = str(self.llama_dir / lib_file)
        self.process = None
        self.verbose = verbose
        self.wait_for_ready = wait_for_ready
        self.subprocess_kwargs = subprocess_kwargs
        debug_logger.debug(f'LlamaBaseServer init, server_url: {self.server_url}')


    @staticmethod
    def is_jupyter_runtime():
        """Checking that the runtime environment is Jupyter or Colab"""
        return (
            'google.colab' in sys.modules or
            'ipykernel' in sys.modules or
            'jupyter' in sys.modules
        )


    def log_output_pty(self) -> None:
        """Read and forward llama.cpp server output from a pseudo-terminal (PTY).

        Used in Jupyter/Colab environments to preserve TTY semantics required for
        dynamic progress bar rendering, which is disabled when stdout is a PIPE.
        """
        state = dict(buffer=b'', last_was_cr=False)
        while True:
            try:
                chunk = os.read(self.pty_master_fd, 1)
            except OSError:
                break
            if not chunk:
                break
            self.process_log_output_chunk(chunk, state, '')
