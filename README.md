

# llama-cpp-py

[![PyPI version](https://img.shields.io/pypi/v/llama-cpp-py?color=006400)](https://pypi.org/project/llama-cpp-py/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llama-cpp-py?color=006400&logo=python&logoColor=gold)](https://pypi.org/project/llama-cpp-py/)

Python wrapper for running the [llama.cpp](https://github.com/ggml-org/llama.cpp) server with automatic or manual binary management.  
Runs the server in a separate subprocess supporting both synchronous and asynchronous APIs.


## Requirements

Python 3.10 or higher.


## Installation

From PyPI
```sh
pip install llama-cpp-py
```

From source
```sh
git clone https://github.com/sergey21000/llama-cpp-py
cd llama-cpp-py
pip install -e .
```

Using [UV](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)
```sh
uv pip install llama-cpp-py
```


## Quick Start

More examples in the Google Colab notebook <a href="https://colab.research.google.com/drive/17f6tD5TM9EP52-3NZtZ1qQ-QrrLUTBEG"><img src="https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=b3771e&label=%20" alt="Open in Colab"></a>


### 1. Set up environment file for llama.cpp

Creating an `env.llama` file with variables for llama.cpp server
```sh
# download example env file
wget https://github.com/sergey21000/llama-cpp-py/raw/main/env.llama
# or create manually
nano env.llama
```

Example `env.llama` content:
```env
# llama.cpp server environment variables
# See: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#usage

# Model source
LLAMA_ARG_MODEL_URL=https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf

# Alternative model options
# LLAMA_ARG_MODEL=D:/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
# LLAMA_ARG_HF_REPO=bartowski/Qwen_Qwen3-0.6B-GGUF
# LLAMA_ARG_HF_FILE=Qwen_Qwen3-0.6B-Q4_K_M.gguf

# Server configuration
LLAMA_ARG_JINJA=1
LLAMA_ARG_CTX_SIZE=4096
LLAMA_ARG_NO_WEBUI=1
LLAMA_ARG_N_PARALLEL=1
LLAMA_ARG_N_GPU_LAYERS=-1

# Network endpoint
LLAMA_ARG_PORT=8080
LLAMA_ARG_HOST=127.0.0.1
```


### 2. Launch the server and send requests

Launching a synchronous server based on the latest [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) version
```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_cpp_py import LlamaSyncServer


# environment variables for llama.cpp
load_dotenv(dotenv_path='env.llama')

# auto-download last release and start server
# set verbose=True to display server logs
server = LlamaSyncServer()
server.start(verbose=False)


# sending requests with OpenAI client
client = OpenAI(
	base_url=server.server_url + '/v1',
	api_key='sk-no-key-required',
)
response = client.chat.completions.create(
    model='local',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

# stopping the server
server.stop()
```

Launching an asynchronous server based on a specific release version
```python
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from llama_cpp_py import LlamaAsyncServer, LlamaReleaseManager


# environment variables for llama.cpp
load_dotenv(dotenv_path='env.llama')

# a) download a release by a specific tag with the 'cuda' priority in the title
# set tag='latest' to use the latest llama.cpp release version
# optionally specify priority_patterns to prefer certain builds (e.g. 'cuda' or 'cpu')
release_manager = LlamaReleaseManager(tag='b6780', priority_patterns=['cuda'])

# b) or set a specific release url in zip format
# release_manager = LlamaReleaseManager(
#     release_zip_url='https://github.com/ggml-org/llama.cpp/releases/download/b6780/llama-b6780-bin-win-cuda-12.4-x64.zip'
# )

# c) or selecting the compiled directory llama.cpp
# release_manager = LlamaReleaseManager(release_dir='/content/llama.cpp/build/bin')
	
async def main():
    # start llama.cpp server (set verbose=True to display server logs)
    llama_server = LlamaAsyncServer(verbose=False, release_manager=release_manager)
    await llama_server.start()

    # sending requests with OpenAI client
    client = AsyncOpenAI(
        base_url=f'{llama_server.server_url}/v1',
        api_key='sk-no-key-required',
    )
    stream_response = await client.chat.completions.create(
        model='local',
        messages=[{'role': 'user', 'content': 'How are you?'}],
        stream=True,
        temperature=0.8,
        max_tokens=-1,
        extra_body=dict(
            top_k=40,
            reasoning_format='none',
            chat_template_kwargs=dict(
                enable_thinking=True,
            ),
        ),
    )
    full_response = ''
    async for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            full_response += token
            print(token, end='', flush=True)

    # stopping the server
    await llama_server.stop()

if __name__ == '__main__':
    asyncio.run(main())
```

Use Context manager
```python
import os
from openai import AsyncOpenAI
from llama_cpp_py import LlamaAsyncServer

os.environ['LLAMA_ARG_MODEL_URL'] = 'https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_S.gguf'

async with LlamaAsyncServer() as server:
    client = AsyncOpenAI(
        base_url=f'{server.server_url}/v1',
        api_key='sk-no-key-required',
    )
    stream_response = await client.chat.completions.create(
        model='local',
        messages=[{'role': 'user', 'content': 'Hello!'}],
        stream=True,
    )
    full_response = ''
    async for chunk in stream_response:
        if (token := chunk.choices[0].delta.content) is not None:
            full_response += token
            print(token, end='', flush=True)
```


## Troubleshooting

If the server fails to start or behaves unexpectedly, check the following:
- Check that the model path or URL in `env.llama` is correct
- Verify that the port is not already in use
- Try setting `verbose=True` to see server logs
```python
llama_server = LlamaAsyncServer(verbose=True)
```
- Link to the [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) archive appropriate for your system via 
```python
LlamaReleaseManager(release_zip_url=url)
```
- Or use the path to the directory with the pre-compiled llama.cpp 
```python
LlamaReleaseManager(release_dir=path_to_binaries)
```

---
If the model is being downloaded from a URL and the server times out before it finishes loading, you can:
- Increase the startup timeout by setting the environment variable
```python
import os
os.environ['TIMEOUT_WAIT_FOR_SERVER'] = 600  # default 300
```
(value is in seconds), or

- Pre-download the model manually and set its local path in
```python
import os
os.environ['LLAMA_ARG_MODEL'] = 'C:\path\to\model.gguf'
```

---
llama.cpp binary releases are downloaded to:  
- **Windows**
```
%LOCALAPPDATA%\llama-cpp-py\releases
```
- **Linux**
```
~/.local/share/llama-cpp-py/releases
```
- **MacOS**
```
~/Library/Application Support/llama-cpp-py/releases
```
See [platformdirs examle output](https://github.com/tox-dev/platformdirs?tab=readme-ov-file#example-output)


## License

This project is licensed under the terms of the [MIT](./LICENSE) license.
