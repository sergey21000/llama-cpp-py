from openai import OpenAI, AsyncOpenAI

from llama_cpp_py.utils.llm_formatter import LLMFormatter


class LlamaBaseClient:
    """
    Base client for interacting with LLM models, providing common preprocessing
    and postprocessing utilities for text generation.
    """

    def __init__(
        self,
        openai_base_url: str,
        api_key: str,
        model: str,
        async_mode: bool,
    ):
        """
        Initialize an synchronous/asynchronous client for a llama.cpp server.

        Args:
            openai_base_url: Base URL of the OpenAI-compatible API endpoint.
                Must include the version prefix (e.g., 'http://localhost:8080/v1').
            api_key: API key for authentication. For llama.cpp servers that do not
                require authentication, a placeholder such as '-' can be used.
        """
        if '0.0.0.0' in openai_base_url:
            openai_base_url = openai_base_url.replace('0.0.0.0', '127.0.0.1')
        self.openai_base_url = openai_base_url
        client_class = AsyncOpenAI if async_mode else OpenAI
        self.client = client_class(
            base_url=openai_base_url,
            api_key=api_key,
        )
        self.model = model
        self.formatter = LLMFormatter()
