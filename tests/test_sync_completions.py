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
from openai import OpenAI
from colorama import Fore, Style


def test_sync_completion(llama_sync_server):
    """Test synchronous chat completions with various parameters.
    
    Verifies:
    - Sync client integration works
    - Different parameters affect output as expected
    - Response formatting is correct
    """
    client = OpenAI(
        base_url=f'{llama_sync_server.server_url}/v1',
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
    stream_response = client.chat.completions.create(
        model='local',
        messages=[{'role':'user', 'content': 'Привет, как дела?'}],
        stream=True,
        **chat_completions_kwargs,
    )
    response_text = ''
    for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            response_text += token
            
    print()
    print(f'{Fore.YELLOW}{Style.BRIGHT}response_text:{Style.RESET_ALL}\n{response_text}')
    assert len(response_text.split()) > 1
    assert '<think>' not in response_text

    chat_completions_kwargs['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    stream_response = client.chat.completions.create(
        model='local',
        messages=[{'role':'user', 'content': 'Привет, как дела?'}],
        stream=True,
        **chat_completions_kwargs,
    )
    response_text = ''
    for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            response_text += token
    
    print(f'{Fore.YELLOW}{Style.BRIGHT}response_text:{Style.RESET_ALL}\n{response_text}')
    assert len(response_text.split()) > 1
    assert '<think>' in response_text
