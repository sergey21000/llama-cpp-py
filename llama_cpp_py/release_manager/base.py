import platform
import zipfile
import tarfile
import stat
from pathlib import Path

import requests
from tqdm import tqdm

from llama_cpp_py.logger import logger


class GithubReleaseManager:
    """Manages downloading and extracting GitHub releases for specific platforms.
    
    Handles automatic detection of system architecture, downloading appropriate
    release assets, and extracting them to a local directory with caching support.
    """
    def __init__(
        self,
        releases_api_url: str,
        releases_dir: str | Path,
        tag: str = 'latest',
        release_zip_url: str = '',
        exclude_patterns: list[str] | None = None,
        priority_patterns: list[str] | None = None,
    ):
        """Initialize GitHub release manager.
        
        Args:
            releases_api_url: GitHub API releases endpoint URL
            releases_dir: Local directory to store downloaded releases
            tag: Release tag name or 'latest' for the most recent release
            release_zip_url: Direct URL to specific release zip file (overrides tag)
            exclude_patterns: Patterns to exclude from asset selection
            priority_patterns: Patterns to prioritize when multiple assets match
        """
        self.validate_releases_api_url(releases_api_url)
        self.releases_api_url = releases_api_url
        self.releases_dir = Path(releases_dir)
        self.releases_dir.mkdir(parents=True, exist_ok=True)
        if release_zip_url:
            tag = self.get_tag_name_from_url(release_zip_url)
        elif tag == 'latest':
            tag = self.get_tag_name_from_url(self.releases_api_url + '/latest')
        self.tag = tag
        if not release_zip_url:
            release_zip_url = self.get_release_zip_url(
                tag=self.tag,
                exclude_patterns=exclude_patterns,
                priority_patterns=priority_patterns,
            )
        self.validate_release_zip_url(release_zip_url)
        self.release_dir = self.releases_dir / Path(Path(release_zip_url).stem).stem
        if not self.release_dir.exists():
            self.download_and_extract_zip(
            zip_url=release_zip_url,
            extract_dir=self.release_dir,
        )
        else:
            logger.info(f'Using cached release: {self.release_dir}')


    @staticmethod
    def validate_releases_api_url(releases_api_url):
        """Validate GitHub releases API URL format."""
        # https://api.github.com/repos/ggml-org/llama.cpp/releases
        if not (
            releases_api_url.startswith('https://api.github.com/repos/')
            and releases_api_url.endswith('/releases')
        ):
            raise ValueError(
                'The URL with releases must start with '
                'https://api.github.com/repos/ and end with /releases)'
            )


    @staticmethod
    def validate_release_zip_url(release_zip_url: str) -> None:
        """Validate GitHub release zip URL format."""
        # https://github.com/ggml-org/llama.cpp/releases/download/b6752/cudart-llama-bin-win-cuda-12.4-x64.zip
        if not (
            release_zip_url.startswith('https://github.com/')
            and (release_zip_url.endswith('.zip') or release_zip_url.endswith('.gz'))
        ):
            raise ValueError(
                'The URL with release must start with '
                'https://github.com/ and end with .zip)'
            )


    @staticmethod
    def get_tag_name_from_url(url: str) -> str:
        """Extract release tag name from GitHub API URL or download URL."""
        if url.endswith('/releases/latest') and 'api.github.com' in url:
            response = requests.get(url)
            response.raise_for_status()
            release_data = response.json()
            if not isinstance(release_data, dict):
                raise ValueError(
                    f'Returned a list instead of a dictionary at requests.get("{url}").json().\n'
                    'The URL does not lead to the page of one release'
                )
            tag_name = release_data.get('tag_name')
        else:
            tag_name = url.split('releases/download/')[-1].split('/')[0]
        if not tag_name:
            raise ValueError(f'Tag not found at {url}')
        return tag_name


    def get_release_zip_url(
        self,
        tag: str,
        exclude_patterns: list[str] | None = None,
        priority_patterns: list[str] | None = None,
    ) -> str:
        """Get download URL for the most suitable release zip asset."""
        zip_assets = self.get_release_zip_assets(tag=tag)
        zip_asset = self.get_matched_asset(
            assets=zip_assets,
            exclude_patterns=exclude_patterns,
            priority_patterns=priority_patterns,
        )
        return zip_asset['url']


    def get_release_zip_assets(self, tag: str) -> list[dict[str, str]]:
        """Get all zip assets available for a specific release tag."""
        api_url = f'{self.releases_api_url}/tags/{tag}'
        response = requests.get(api_url)
        response.raise_for_status()
        release_data = response.json()
        zip_assets = []
        for asset in release_data['assets']:
            if asset['name'].endswith('.zip') or asset['name'].endswith('.gz'):
                zip_assets.append({
                    'name': asset['name'],
                    'tag_name': release_data['tag_name'],
                    'url': asset['browser_download_url'],
                    'size': f'{asset["size"] // 1024**2} MB',
                })
        return zip_assets


    def get_matched_asset(
        self,
        assets: list[dict[str, str]],
        exclude_patterns: list[str] | None = None,
        priority_patterns: list[str] | None = None,
    ) -> dict[str, str]:
        """Select the most appropriate asset based on system and patterns."""
        os_name, arch = self.detect_system()
        matched_assets = []
        for asset in assets:
            name = asset['name'].lower()
            if os_name not in name or arch not in name:
                continue
            if exclude_patterns and any(p in name for p in exclude_patterns):
                continue
            matched_assets.append(asset)
        if not matched_assets:
            raise RuntimeError(f'No suitable archive found for {os_name}-{arch}')
        if priority_patterns:
            for pattern in priority_patterns:
                for asset in matched_assets:
                    name = asset['name'].lower()
                    if pattern in name:
                        matched_assets = [asset]
                        break
                if len(matched_assets) == 1:
                    break
        if len(matched_assets) > 1:
            logger.warning(
                f'More than one archive match found, the first one will be selected: '
                f'{[d.get("name") for d in matched_assets]}'
            )
        return matched_assets[0]


    @staticmethod
    def detect_system() -> tuple[str, str]:
        """Detect current operating system and architecture."""
        os_name = platform.system().lower()
        arch = platform.machine().lower()
        if os_name == 'windows':
            os_name = 'win'
        elif os_name == 'linux':
            os_name = 'ubuntu'
        elif os_name == 'darwin':
            os_name = 'macos'
        if arch in ('x86_64', 'amd64'):
            arch = 'x64'
        elif arch in ('arm64', 'aarch64'):
            arch = 'arm64'
        return os_name, arch


    @staticmethod
    def download_file(file_url: str, file_path: str | Path) -> None:
        """Download file from URL with progress bar."""
        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            raise Exception(
                f'The file is not available for download at the link: {file_url}'
            )
        total_size = int(response.headers.get('content-length', 0))
        progress_tqdm = tqdm(
            desc=f'Downoading release: {Path(file_path).name}',
            total=total_size,
            unit='iB',
            unit_scale=True,
        )
        with open(file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=4096):
                size = file.write(data)
                progress_tqdm.update(size)
        progress_tqdm.close()


    @classmethod
    def extract_archive(cls, zip_or_tar_path: Path, extract_dir: Path) -> None:
        """Extract .zip or .tar(.gz) archive into extract_dir."""
        #  TAR | .GZ
        if zip_or_tar_path.suffix in ['.tar', '.gz', '.tar.gz', '.tgz']:
            with tarfile.open(zip_or_tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            return
        # ZIP
        if zip_or_tar_path.suffix == '.zip':
            return cls._extract_zip_with_symlinks(
                zip_path=zip_or_tar_path, extract_dir=extract_dir,
            )
        raise ValueError(f'Unsupported archive type: {zip_or_tar_path}')


    @staticmethod
    def _extract_zip_with_symlinks(zip_path: Path, extract_dir: Path) -> None:
        """Extract ZIP while manually restoring symbolic links (ZIP doesn't preserve them)."""
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for info in archive.infolist():
                target_path = extract_dir / info.filename
                perms = info.external_attr >> 16
                if stat.S_ISLNK(perms):
                    link_target = archive.read(info).decode()
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if target_path.exists() or target_path.is_symlink():
                        target_path.unlink()
                    os.symlink(link_target, target_path)
                else:
                    archive.extract(info, extract_dir)


    @classmethod
    def download_and_extract_zip(
        cls,
        zip_url: str,
        extract_dir: Path,
        override: bool = False,
        set_execute_permissions: bool = True,
    ) -> None:
        """Download and extract zip file, optionally setting execute permissions."""
        extract_dir.mkdir(exist_ok=True, parents=True)
        zip_path = extract_dir / Path(zip_url).name
        logger.info(f'Loading file {zip_url} to path {zip_path}')
        cls.download_file(file_url=zip_url, file_path=zip_path)
        cls.extract_archive(zip_or_tar_path=zip_path, extract_dir=extract_dir)
        zip_path.unlink(missing_ok=True)
        if set_execute_permissions and platform.system() != 'Windows':
            for file in extract_dir.rglob('*'):
                if file.is_file():
                    file.chmod(0o755)
