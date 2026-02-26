import asyncio
import pprint
import json
from pathlib import Path
from typing import AsyncIterator, Any

import aiohttp
from openai import AsyncOpenAI

from llama_cpp_py.logger import debug_logger
from llama_cpp_py.client.base import LlamaBaseClient


class LlamaAsyncClient(LlamaBaseClient):
    """
    Asynchronous client for a llama.cpp server exposing an OpenAI-compatible API.
    
    Supports streaming chat completions and optional thinking-mode control.
    """
    
    def __init__(self, openai_base_url: str, api_key: str = '-'):
        """
        Initialize an asynchronous client for a llama.cpp server.
        
        Args:
            openai_base_url: Base URL of the OpenAI-compatible API endpoint.
                Must include the version prefix (e.g., 'http://localhost:8080/v1').
            api_key: API key for authentication. For llama.cpp servers that do not
                require authentication, a placeholder such as '-' can be used.
        """
        if '0.0.0.0' in openai_base_url:
            openai_base_url = openai_base_url.replace('0.0.0.0', '127.0.0.1')
        self.openai_base_url = openai_base_url
        self.client = AsyncOpenAI(
            base_url=openai_base_url,
            api_key=api_key,
        )


    async def check_health(self) -> dict[str, Any] | None:
        """Check llama.cpp server health status asynchronously."""
        return await self._get_request('/health')


    async def get_models(self) -> dict[str, Any] | None:
        """Get list of available models asynchronously."""
        return await self._get_request('/models')


    async def get_props(self) -> dict[str, Any] | None:
        """Retrieve server global properties from Llama.cpp server asynchronously."""
        return await self._get_request('/props')


    async def _get_request(self, path: str) -> dict[str, Any] | None:
        """
        Make an asynchronous GET request to the specified API endpoint.
        
        Args:
            path: API endpoint path (e.g., '/health', '/models')
            
        Returns:
            Response JSON as dict on success, None on failure.
            Failures are logged via debug_logger.
            
        Note:
            Uses a new aiohttp session for each request. If you make many requests,
            consider sharing a session for better performance.
        """
        url = f'{self.openai_base_url}{path}'
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            debug_logger.debug(f'Failed to fetch `{url}`: {e}')
        except json.JSONDecodeError as e:
            debug_logger.debug(f'Invalid JSON response from `{url}`: {e}')
        except asyncio.TimeoutError as e:
            debug_logger.debug(f'Request timeout for `{url}`: {e}')


    async def check_multimodal_support(self, modality: str = 'vision') -> bool:
        """Checking server multimodality support"""
        props = await self.get_props()
        if props:
            return props.get('modalities', {}).get(modality, False)
        return False
        

    async def _astream_chat_completion_tokens(
        self,
        messages: list[dict],
        completions_kwargs: dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Stream tokens from OpenAI Chat Completions API.
        
        Internal method that handles streaming from the legacy/completions API endpoint.
        Creates a streaming request and yields individual content tokens as they arrive.
        
        Args:
            messages: List of message dictionaries in OpenAI format
                    (e.g., [{"role": "user", "content": "Hello"}])
            completions_kwargs: Additional parameters for the completions API
                              (temperature, max_tokens, top_p, etc.)
        
        Yields:
            Individual content tokens as strings from the streaming response.
            Empty tokens are filtered out automatically.
            
        Note:
            This method is designed for internal use by the public stream() method.
            Uses the chat.completions.create endpoint with stream=True.
        """
        stream_response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **completions_kwargs,
        )
        async for chunk in stream_response:
            token = chunk.choices[0].delta.content
            if token:
                yield token


    async def _astream_responses_tokens(
        self,
        input: str | list[dict],
        responses_kwargs: dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Stream tokens from OpenAI Responses API.
        
        Internal method that handles streaming from the newer responses API endpoint.
        Processes the response chunks and extracts delta content for token streaming.
        
        Args:
            input: Either a string (simple prompt) or a list of message dictionaries
                  (formatted conversation history)
            responses_kwargs: Additional parameters for the responses API
                            (temperature, max_output_tokens, instructions, etc.)
        
        Yields:
            Individual content tokens as strings from the streaming response.
            Automatically filters out empty tokens.
            
        Note:
            This method is designed for internal use by the public stream() method.
            Uses the responses.create endpoint with stream=True.
            The responses API uses a different chunk structure than completions API.
        """
        stream_response = await self.client.responses.create(
            model=self.model,
            input=input,
            stream=True,
            **responses_kwargs,
        )
        async for chunk in stream_response:
            token = getattr(chunk, 'delta', '')
            if token:
                yield token


    async def astream(
        self,
        user_message_or_messages: str,
        system_prompt: str = '',
        image_path_or_base64: str | Path = '',
        resize_size: int | None = None,
        show_thinking: bool = True,
        return_per_token: bool = True,
        out_token_in_thinking_mode: str | None = 'Thinking ...',
        use_responses_api: bool = False,
        completions_kwargs: dict[str, Any] = {},
        responses_kwargs: dict[str, Any] = {},
    ) -> AsyncIterator[str]:
        """
        Stream LLM responses with configurable thinking mode and token handling.
        
        High-level generator that provides a unified streaming interface for both
        Chat Completions and Responses APIs. Handles message preparation, thinking
        tag processing, and token accumulation with flexible output options.
        
        Args:
            user_message_or_messages: 
                User input - either a string message or pre-formatted message list.
                When using Responses API with string input, it's treated as a simple prompt.
            
            system_prompt: 
                System instructions to set model behavior and context.
                Ignored if user_message_or_messages contains system messages.
            
            image_path_or_base64: 
                Optional image for multimodal queries. Accepts file path or base64 string.
                When provided, automatically formats messages for vision capabilities.
            
            resize_size: 
                Maximum image dimension in pixels (default 512). Images are resized
                maintaining aspect ratio to optimize token usage while preserving detail.
            
            show_thinking: 
                Control visibility of  tags content. If True, includes thinking
                content in output. If False, filters out thinking sections and shows
                placeholder instead.
            
            return_per_token: 
                Streaming granularity. If True, yields individual tokens as they arrive.
                If False, accumulates tokens and yields complete sentences/text blocks.
            
            out_token_in_thinking_mode: 
                Placeholder text shown when thinking content is hidden (show_thinking=False).
                Set to None to show nothing during thinking. Default shows "Thinking ...".
            
            use_responses_api: 
                API selection flag. If True, uses the newer Responses API endpoint.
                If False, uses the legacy Chat Completions API. Affects which kwargs
                dictionary should be provided.
            
            completions_kwargs: 
                Additional parameters for Chat Completions API when use_responses_api=False.
                Common options: temperature, max_tokens, top_p, presence_penalty.
                Must be empty when using Responses API.
            
            responses_kwargs: 
                Additional parameters for Responses API when use_responses_api=True.
                Common options: temperature, max_output_tokens, instructions, metadata.
                Must be empty when using Completions API.
        
        Yields:
            Processed text chunks based on configuration. When return_per_token=True,
            yields individual tokens. When return_per_token=False, yields accumulated
            text blocks. Empty strings are filtered out.
        """
        if (
            use_responses_api and completions_kwargs
        ) or (
            not use_responses_api and responses_kwargs
        ):
            debug_logger.warning(
                'When using use_responses_api=True, you must pass responses_kwargs;\n'
                'When using use_responses_api=False, you must pass completions_kwargs.\n' 
                'Skipping sending a request'
            )
            return
        messages = self._prepare_messages(
            user_message_or_messages=user_message_or_messages,
            system_prompt=system_prompt,
            image_path_or_base64=image_path_or_base64,
            resize_size=resize_size,
            use_responses_api=use_responses_api,
        )
        if not messages:
            debug_logger.warning(
                'Messages list is empty. Request will not be sent to the server.'
            )
            return
        debug_logger.debug(
            f'Messages before openai chat.completions.create:\n{pprint.pformat(messages)}'
        )
        if use_responses_api:
            generator = self._stream_responses_tokens(
                input=messages,
                responses_kwargs=responses_kwargs,
            )
        else:
            generator = self._stream_chat_completion_tokens(
                messages=messages,
                completions_kwargs=completions_kwargs,
            )
        state = dict(response_text='', is_in_thinking=False)
        async for token in generator:
            token = self._process_output_token(
                token=token,
                state=state,
                show_thinking=show_thinking,
                return_per_token=return_per_token,
                out_token_in_thinking_mode=out_token_in_thinking_mode,
            )
            if token:
                yield token
