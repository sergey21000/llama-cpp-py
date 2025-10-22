import os
import asyncio
import pytest
import pytest_asyncio

from dotenv import load_dotenv

from llama_cpp_py import (
    LlamaReleaseManager,
    LlamaSyncServer,
    LlamaAsyncServer,
)


@pytest.fixture(scope='session')
def llama_env():
    """Load environment variables for llama.cpp server from env.llama file."""
    load_dotenv('env.llama')


@pytest.fixture(scope='session')
def release_manager():
    """Create LlamaReleaseManager instance with specific release tag.
    
    Returns:
        Configured LlamaReleaseManager for testing
    """
    return LlamaReleaseManager(tag='b6780')


@pytest.fixture
def llama_sync_server(release_manager, llama_env):
    """Fixture for synchronous llama.cpp server instance.
    
    Starts server before test, yields server instance, stops after test.
    """
    server = LlamaSyncServer(verbose=False, release_manager=release_manager)
    server.start()
    yield server
    server.stop()


@pytest_asyncio.fixture
async def llama_async_server(release_manager, llama_env):
    """Fixture for asynchronous llama.cpp server instance.
    
    Starts server before test, yields server instance, stops after test.
    """
    server = LlamaAsyncServer(verbose=False, release_manager=release_manager)
    await server.start()
    yield server
    await server.stop()
