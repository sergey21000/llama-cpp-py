import os
import asyncio
import pytest
import pytest_asyncio

from dotenv import dotenv_values

from llama_cpp_py import (
    LlamaReleaseManager,
    LlamaSyncServer,
    LlamaAsyncServer,
)


@pytest.fixture(scope='session')
def llama_env():
    env = dotenv_values('env.llama')
    env.update(os.environ)
    return env


@pytest.fixture(scope='session')
def release_manager():
    return LlamaReleaseManager(tag='b6780')


@pytest.fixture
def llama_sync_server(release_manager, llama_env):
    # print("DEBUG: llama_env in fixture:", llama_env)
    server = LlamaSyncServer(verbose=False, release_manager=release_manager)
    server.start(env=llama_env)
    yield server
    server.stop()


@pytest_asyncio.fixture
async def llama_async_server(release_manager, llama_env):
    server = LlamaAsyncServer(verbose=False, release_manager=release_manager)
    await server.start(env=llama_env)
    yield server
    await server.stop()
