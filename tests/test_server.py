"""
Integration tests for llama-cpp-py package.

Tests cover:
- Server startup/shutdown (sync and async)
"""

from llama_cpp_py import LlamaSyncClient

def test_server_startup(llama_sync_server):
    health = LlamaSyncClient(openai_base_url=llama_sync_server.openai_base_url).check_health()
    assert health.get('ok', False)
