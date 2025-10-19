import os
import platform
from pathlib import Path

from platformdirs import user_data_dir

from llama_cpp_py.logger import logger
from llama_cpp_py.release_manager.base import GithubReleaseManager


class LlamaReleaseManager(GithubReleaseManager):
    def __init__(
        self,
        tag: str = 'latest',
        release_zip_url: str = '',
        exclude_patterns: list[str] | None = ['vulkan', 'cudart'],
        priority_patterns: list[str] | None = ['cpu', 'cuda'],
    ):
        releases_api_url = 'https://api.github.com/repos/ggml-org/llama.cpp/releases'
        releases_dir = Path(user_data_dir('llama-cpp-py', appauthor=False)) / 'releases'
        releases_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            releases_api_url=releases_api_url,
            releases_dir=releases_dir,
            tag=tag,
            release_zip_url=release_zip_url,
            exclude_patterns=exclude_patterns,
            priority_patterns=priority_patterns,
        )
        self.ensure_release_dir(self.release_dir)
        if platform.system() != 'Windows':
            os.environ['LD_LIBRARY_PATH'] = (
                f"{os.getenv('LD_LIBRARY_PATH') + ':' if os.getenv('LD_LIBRARY_PATH') else ''}"
                f"{self.release_dir.absolute()}"
            )

    def ensure_release_dir(self, release_dir: Path) -> None:
        if self.validate_release_dir(release_dir):
            return
        if platform.system() != 'Windows':
            bin_dir = release_dir / 'build' / 'bin'
            if self.validate_release_dir(bin_dir):
                self.release_dir = bin_dir
            else:
                raise ValueError(f'llama-server not found in {release_dir} or {bin_dir}')

    def validate_release_dir(self, release_dir: Path) -> None:
        if not any(p.stem == 'llama-server' for p in release_dir.iterdir()):
            return False
        return True
            
    @staticmethod
    def validate_release_zip_url(release_zip_url: str):
        GithubReleaseManager.validate_release_zip_url(release_zip_url)
        if 'https://github.com/ggml-org/llama.cpp/releases/download' not in release_zip_url:
            raise ValueError('The zip download link must include https://github.com/ggml-org/llama.cpp')
