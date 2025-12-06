from typing import AsyncIterator, Any

from openai import AsyncOpenAI

from llama_cpp_py.logger import logger, status_logger
from llama_cpp_py.client.base import LlamaBaseClient


class LlamaAsyncClient(LlamaBaseClient):
    """
    Asynchronous client for llama.cpp server using OpenAI-compatible API.
    
    Provides streaming chat completions with thinking mode control.
    """
    
    def __init__(self, server_url: str):
        """
        Initialize async client for llama.cpp server.
        
        Args:
            server_url: Base URL of the llama.cpp server (e.g., 'http://localhost:8080')
        """
        self.server_url = server_url
        self.client = AsyncOpenAI(
            base_url=f'{server_url}/v1',
            api_key='sk-no-key-required',
        )


    async def astream_chat_completion_tokens(
        self,
        user_message_or_messages: str,
        system_prompt: str,
        completions_kwargs: dict[str, Any],
    ):
        """
        Low-level async generator for raw LLM tokens.
        
        Args:
            user_message_or_messages: User input or pre-formatted messages.
            system_prompt: System instructions.
            completions_kwargs: Additional parameters for completion API.
            
        Yields:
            Raw tokens from the LLM stream.
        """
        messages = self._prepare_messages(
            user_message_or_messages=user_message_or_messages,
            system_prompt=system_prompt,
        )
        logger.debug(f'messages before openai chat.completions.create {messages}')
        stream_response = await self.client.chat.completions.create(
            model='local',
            messages=messages,
            stream=True,
            **completions_kwargs,
        )
        async for chunk in stream_response:
            if (token := chunk.choices[0].delta.content) is not None:
                yield token


    async def astream(
        self,
        user_message_or_messages: str,
        completions_kwargs: dict[str, Any],
        system_prompt: str = '',
        show_thinking: bool = True,
        return_per_token: bool = True,
        out_token_in_thinking_mode: str | None = 'Thinking ...',
    ) -> AsyncIterator[str]:
        """
        High-level async generator with thinking mode control.
        
        Args:
            user_message_or_messages: User input or pre-formatted messages.
            completions_kwargs: Additional parameters for completion API.
            system_prompt: System instructions.
            show_thinking: Whether to include thinking tags in output.
            return_per_token: Yield tokens individually or accumulated text.
            out_token_in_thinking_mode: Placeholder for thinking content.
            
        Yields:
            Processed tokens or text chunks based on configuration.
        """
        generator = self.astream_chat_completion_tokens(
            user_message_or_messages=user_message_or_messages,
            system_prompt=system_prompt,
            completions_kwargs=completions_kwargs,
        )
        state = dict(response_text='', is_in_thinking=False)
        async for token in generator:
            token = self.process_output_token(
                token=token,
                state=state,
                show_thinking=show_thinking,
                return_per_token=return_per_token,
                out_token_in_thinking_mode=out_token_in_thinking_mode,
            )
            if token:
                yield token
