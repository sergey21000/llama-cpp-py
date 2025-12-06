import os
import platform
from pathlib import Path

from platformdirs import user_data_dir

from llama_cpp_py.logger import logger
from llama_cpp_py.release_manager.base import GithubReleaseManager


class LlamaReleaseManager(GithubReleaseManager):
    """Specialized release manager for llama.cpp binaries.
    
    Handles automatic setup of llama-server binaries with platform-specific
    optimizations and environment configuration.
    """
    def __init__(
        self,
        tag: str = 'latest',
        release_zip_url: str = '',
        releases_dir: str | Path = '',
        exclude_patterns: list[str] | None = ['vulkan', 'cudart'],
        priority_patterns: list[str] | None = ['cpu', 'cuda'],
    ):
        """Initialize llama.cpp release manager.
        
        Args:
            tag: Release tag name or 'latest' for the most recent release
            release_zip_url: Direct URL to specific llama.cpp release zip
            releases_dir: Local directory for releases (uses appdata if not specified)
            exclude_patterns: Patterns to exclude from asset selection
            priority_patterns: Patterns to prioritize when multiple assets match
        """
        releases_api_url = 'https://api.github.com/repos/ggml-org/llama.cpp/releases'
        if not releases_dir:
            releases_dir = os.getenv('RELEASES_DIR')
            if not releases_dir:
                releases_dir = Path(
                    user_data_dir('llama-cpp-py', appauthor=False)
                ) / 'releases'
        super().__init__(
            releases_api_url=releases_api_url,
            releases_dir=releases_dir,
            tag=tag,
            release_zip_url=release_zip_url,
            exclude_patterns=exclude_patterns,
            priority_patterns=priority_patterns,
        )
        self.ensure_release_dir(release_dir=self.release_dir, tag=self.tag)
        if platform.system() != 'Windows':
            os.environ['LD_LIBRARY_PATH'] = f"{self.release_dir.absolute()}:" + ':'.join(
                p for p in os.getenv('LD_LIBRARY_PATH', '').split(':') 
                if 'llama-cpp-py/releases' not in p
            )

    def ensure_release_dir(self, release_dir: Path, tag: str) -> None:
        """Ensure the release directory contains valid llama-server binaries."""
        if self.validate_release_dir(release_dir):
            return
        if platform.system() != 'Windows':
            bin_dir = release_dir / 'build' / 'bin'
            if self.validate_release_dir(bin_dir):
                self.release_dir = bin_dir
                return
        bin_dir = release_dir / f'llama-{tag}'
        if self.validate_release_dir(bin_dir):
            self.release_dir = bin_dir
            return
        raise ValueError(f'llama-server not found in {release_dir} or {bin_dir}')

    def validate_release_dir(self, release_dir: Path) -> None:
        """Validate that release directory contains llama-server executable."""
        if not release_dir.exists():
            logger.debug(f'Path {release_dir} not exists')
            return False
        if not any(p.stem == 'llama-server' for p in release_dir.iterdir()):
            return False
        return True
            
    @staticmethod
    def validate_release_zip_url(release_zip_url: str):
        """Validate llama.cpp specific release zip URL format."""
        GithubReleaseManager.validate_release_zip_url(release_zip_url)
        if 'https://github.com/ggml-org/llama.cpp/releases/download' not in release_zip_url:
            raise ValueError('The zip download link must include https://github.com/ggml-org/llama.cpp')

