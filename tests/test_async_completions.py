"""
Integration tests for llama-cpp-py package.

Tests cover:
- Server startup/shutdown (sync and async)
- OpenAI client integration
- Chat completions with streaming
- Thinking mode functionality
- Environment configuration
"""

import os
import pytest
from openai import AsyncOpenAI
from colorama import Fore, Style


@pytest.mark.asyncio
async def test_async_completion(llama_async_server):
    """Test async chat completions with and without thinking mode.
    
    Verifies:
    - Basic completion works
    - Response contains reasonable text
    - Thinking tags are properly controlled via enable_thinking flag
    """
    client = AsyncOpenAI(
        base_url=f'{llama_async_server.server_url}/v1',
        api_key='sk-no-key-required',
    )
    chat_completions_kwargs = dict(
        temperature=0.8,
        top_p=0.9,
        max_tokens=5,
        extra_body=dict(
            top_k=40,
            repeat_penalty=1,
            reasoning_format='none',
            chat_template_kwargs=dict(
                enable_thinking=False,
            ),
        ),
    )
    stream_response = await client.chat.completions.create(
        model='local',
        messages=[{'role':'user', 'content': 'Привет, как дела?'}],
        stream=True,
        **chat_completions_kwargs,
    )
    response_text = ''
    async for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            response_text += token

    print()
    print(f'{Fore.YELLOW}{Style.BRIGHT}response_text:{Style.RESET_ALL}\n{response_text}')
    assert len(response_text.split()) > 1
    assert '<think>' not in response_text

    chat_completions_kwargs['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    stream_response = await client.chat.completions.create(
        model='local',
        messages=[{'role':'user', 'content': 'Привет, как дела?'}],
        stream=True,
        **chat_completions_kwargs,
    )
    response_text = ''
    async for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            response_text += token

    print(f'{Fore.YELLOW}{Style.BRIGHT}response_text:{Style.RESET_ALL}\n{response_text}')
    assert len(response_text.split()) > 1
    assert '<think>' in response_text
