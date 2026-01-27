from pathlib import Path
from typing import Iterator, Any

import requests
from openai import OpenAI

from llama_cpp_py.logger import logger, status_logger
from llama_cpp_py.client.base import LlamaBaseClient


class LlamaSyncClient(LlamaBaseClient):
    """
    Synchronous client for a llama.cpp server exposing an OpenAI-compatible API.
    
    Supports streaming chat completions and optional thinking-mode control.
    """
    
    def __init__(self, openai_base_url: str, api_key: str = '-'):
        """
        Initialize an synchronous client for a llama.cpp server.
        
        Args:
            openai_base_url: Base URL of the OpenAI-compatible API endpoint.
                Must include the version prefix (e.g., 'http://localhost:8080/v1').
            api_key: API key for authentication. For llama.cpp servers that do not
                require authentication, a placeholder such as '-' can be used.
        """
        self.openai_base_url = openai_base_url
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=api_key,
        )


    def check_health(self, path: str = '/health') -> dict[str, str | int]:
        """
        Check llama.cpp server health.

        Returns:
            {
                "ok": True/False,
                "status": "ready" | "loading" | "unavailable" | "down",
                "message": Optional[str]
            }
        """
        url = f'{self.openai_base_url}{path}'
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                return {'ok': True, 'code': resp.status_code, 'status': 'ready'}
            if resp.status_code == 503:
                data = resp.json()
                msg = data.get('error', {}).get('message', 'Loading')
                return {
                    'ok': False,
                    'code': resp.status_code,
                    'status': 'loading',
                    'message': msg,
                }
            text = resp.text
            return {
                'ok': False,
                'code': resp.status_code,
                'status': 'unavailable',
                'message': f'HTTP {resp.status_code}: {text}',
            }
        except Exception as e:
            return {
                'ok': False,
                'code': -1,
                'status': 'down',
                'message': str(e),
            }


    def _stream_chat_completion_tokens(
        self,
        user_message_or_messages: str | list[dict],
        system_prompt: str,
        image_path_or_base64: str | Path,
        resize_size: int,
        completions_kwargs: dict[str, Any],
    ) -> Iterator[str]:
        """
        Low-level sync generator for raw multimodal LLM tokens.
        
        Handles image preprocessing, message formatting, and direct streaming
        from the OpenAI-compatible llama.cpp server API.
        
        Args:
            user_message_or_messages: User text input or pre-formatted messages.
            system_prompt: System instructions for the model.
            image_path_or_base64: Optional image input (file path or base64).
            resize_size: Image resizing dimension for token optimization.
            completions_kwargs: Additional parameters for chat completion API.
            
        Yields:
            Raw tokens from the LLM stream response.
            
        Note:
            Images are automatically resized and converted to base64 format.
            Uses the server's '/v1/chat/completions' endpoint with streaming.
        """
        messages = self._prepare_messages(
            user_message_or_messages=user_message_or_messages,
            system_prompt=system_prompt,
            image_path_or_base64=image_path_or_base64,
            resize_size=resize_size,
        )
        logger.debug(f'Messages before openai chat.completions.create {messages}')
        stream_response = self.client.chat.completions.create(
            model='local',
            messages=messages,
            stream=True,
            **completions_kwargs,
        )
        for chunk in stream_response:
            if (token := chunk.choices[0].delta.content) is not None:
                yield token


    def stream(
        self,
        user_message_or_messages: str,
        system_prompt: str = '',
        image_path_or_base64: str | Path = '',
        resize_size: int = 512,
        show_thinking: bool = True,
        return_per_token: bool = True,
        out_token_in_thinking_mode: str | None = 'Thinking ...',
        completions_kwargs: dict[str, Any] = {},
    ) -> Iterator[str]:
        """
        High-level sync generator for multimodal LLM responses with thinking control.
        
        Provides a convenient interface for streaming LLM responses with
        configurable thinking mode handling, token accumulation, and image support.
        
        Args:
            user_message_or_messages: User text input or pre-formatted messages.
            system_prompt: System instructions for the model.
            image_path_or_base64: Optional image input for multimodal queries.
            resize_size: Maximum image dimension (default 512px).
            show_thinking: Include thinking tags content in output if True.
            return_per_token: Stream individual tokens if True, accumulated text if False.
            out_token_in_thinking_mode: Placeholder text for thinking mode when hidden.
            completions_kwargs: Additional API parameters (temperature, max_tokens, etc.).
            
        Yields:
            Processed tokens or text chunks based on configuration.
            
        Example:
            for token in client.stream("Describe this image", image_path_or_base64="photo.jpg"):
                print(token, end="", flush=True)
                
        Note:
            Default image size (512px) balances visual detail with token efficiency.
            Thinking mode placeholder appears once when entering thinking tags.
        """
        generator = self._stream_chat_completion_tokens(
            user_message_or_messages=user_message_or_messages,
            system_prompt=system_prompt,
            completions_kwargs=completions_kwargs,
            image_path_or_base64=image_path_or_base64,
            resize_size=resize_size,
        )
        state = dict(response_text='', is_in_thinking=False)
        for token in generator:
            token = self._process_output_token(
                token=token,
                state=state,
                show_thinking=show_thinking,
                return_per_token=return_per_token,
                out_token_in_thinking_mode=out_token_in_thinking_mode,
            )
            if token:
                yield token
