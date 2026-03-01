import io
import base64
from pathlib import Path

from PIL import Image

from llama_cpp_py.logger import debug_logger


GRADIO_CHAT_HISTORY = list[dict[str, str | list]]


class LLMFormatter:
    """
    Format messages and process outputs for LLM interactions.
    
    Provides utilities for preparing inputs and processing outputs
    in formats compatible with various LLM APIs (OpenAI, etc.)
    """

    opening_thinking_tags = ['<think>', '&lt;think&gt;']
    closing_thinking_tags = ['</think>', '&lt;/think&gt;']
    all_thinking_tags = [*opening_thinking_tags, *closing_thinking_tags]
    image_extension = ['.png', '.jpg', '.jpeg', '.webp']

    @classmethod
    def prepare_messages(
        cls,
        user_message_or_messages: str | list[dict],
        system_prompt: str,
        image_path: str | Path,
        resize_size: int | None,
        use_responses_api: bool,
        support_system_role: bool = True,
    ) -> list[dict]:
        """
        Prepare messages for multimodal LLM input with optional image support.

        Formats user text, system prompt, and optional image into the OpenAI
        chat completion message format. Supports both text-only and image-text
        multimodal inputs.

        Args:
            user_message_or_messages: User text input or pre-formatted message list.
            system_prompt: System instructions for the model.
            image_path: Path to image file.
            resize_size: Maximum dimension for image resizing (maintains aspect ratio).
            support_system_role: Whether to include system role in messages.

        Returns:
            List of message dictionaries in OpenAI format, with image data as base64.

        Note:
            Images are resized to reduce token usage and converted to base64 PNG.
        """
        if isinstance(user_message_or_messages, list):
            return user_message_or_messages
        messages = []
        if support_system_role and system_prompt:
            messages.append(dict(role='system', content=system_prompt))
        if not image_path:
            text_message = cls._create_message_from_text(
                text=user_message_or_messages,
                use_responses_api=use_responses_api,
            )
            messages.append(text_message)
            return messages
        image_message = cls._create_message_from_image(
            text=user_message_or_messages,
            image_path=image_path,
            resize_size=resize_size,
            use_responses_api=use_responses_api,
        )
        if image_message:
            messages.append(image_message)
        return messages

    @classmethod
    def _prepare_image_to_base64(
        cls,
        image_path: str | Path,
        resize_size: int | None
    ) -> str | None:
        """
        Prepare image for LLM input by resizing and converting to base64.

        Supports image file paths (PNG, JPG, JPEG) or already base64 encoded strings.
        Images are converted to RGB, resized maintaining aspect ratio, and encoded
        as base64 PNG for consistent format.

        Args:
            image: Path to image file or base64 encoded image string.
            resize_size: Maximum width/height for resizing (smaller dimension preserved).

        Returns:
            Base64 encoded PNG image string.

        Raises:
            FileNotFoundError: If image path doesn't exist.
            ValueError: If image format is unsupported.
        """
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)
            if image_path.suffix.lower() in cls.image_extension:
                image_pil = Image.open(image_path).convert('RGB')
                if resize_size:
                    image_pil.thumbnail((resize_size, resize_size))
                buffer = io.BytesIO()
                image_pil.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                return image_base64
            else:
                debug_logger.warning(
                    f'Image format {image_path.suffix} is not supported. ' 
                    f'Expected one of: {cls.image_extension}'
                )
        else:
            debug_logger.warning(
                f'Image must be a string or path, got: {type(image)}' 
            )

    @classmethod
    def _create_message_from_text(
        cls,
        text: str,
        use_responses_api: bool,
        role: str = 'user',
    ) -> dict:
        """
        Create a formatted message dictionary for text-only requests.
        
        Args:
            text: User text input
            use_responses_api: Determines which API format to use
            role: Message role (user, assistant, system)
            
        Returns:
            Dictionary with 'role' and 'content' fields formatted for the specified API
        """
        message_type = 'input_text' if use_responses_api else 'text'
        return dict(
            role=role,
            content=[dict(type=message_type, text=text)],
        )

    @staticmethod
    def _create_message_content_from_text(
        text: str,
        use_responses_api: bool,
    ) -> dict:
        """
        Create a content block for text within a multimodal message.
        
        Args:
            text: Text content
            use_responses_api: Determines which API format to use
            
        Returns:
            Dictionary with text content formatted for the specified API
        """
        if use_responses_api:
            content = dict(type='input_text', text=text)
        else:
            content = dict(type='text', text=text)
        return content

    @classmethod
    def _create_message_content_from_image(
        cls,
        image_path: str,
        resize_size: int,
        use_responses_api: bool,
    ) -> dict | None:
        """
        Create a content block for an image within a multimodal message.
        
        Args:
            image_path: Path to image file
            resize_size: Maximum dimension for image resizing
            use_responses_api: Determines which API format to use
            
        Returns:
            Dictionary with image content formatted for the specified API,
            or None if image preparation fails
        """
        image_base64 = cls._prepare_image_to_base64(
            image_path=image_path,
            resize_size=resize_size,
        )
        if image_base64:
            image_url = f'data:image/png;base64,{image_base64}'
            if use_responses_api:
                content = dict(type='input_image', image_url=image_url)
            else:
                content = dict(
                    type='image_url',
                    image_url=dict(url=image_url),
            )
            return content

    @classmethod
    def _create_message_from_image(
        cls,
        image_path: str | Path,
        resize_size: int | None,
        use_responses_api: bool,
        text: str = '',
        role: str = 'user',
    ) -> dict | None:
        """
        Create a formatted message dictionary for multimodal requests.

        Prepares an image and text combination in the format expected by either
        the Responses API or Chat Completions API.

        Args:
            image_path: Path to image file
            resize_size: Optional maximum dimension for image resizing
            use_responses_api: Determines which API format to use
            text: Optional accompanying text message

        Returns:
            Dictionary with 'role' and 'content' fields formatted for the
            specified API (Responses or Completions).

        Note:
            Returns None if image preparation fails (handled by _prepare_image)
        """
        message = dict(role=role, content=[])
        image_content = cls._create_message_content_from_image(
            image_path=image_path,
            resize_size=resize_size,
            use_responses_api=use_responses_api,
        )
        if image_content:
            message['content'].append(image_content)
            if text:
                text_content = cls._create_message_content_from_text(
                    text=text,
                    use_responses_api=use_responses_api,
                )
                message['content'].append(text_content)
            return message

    
    @classmethod
    def process_output_token(
        cls,
        token: str,
        state: dict,
        show_thinking: bool,
        return_per_token: bool,
        out_token_in_thinking_mode: str,
    ):
        """
        Process individual LLM output token with thinking mode and accumulation control.

        Handles special thinking tags (<think>...</think>) and provides flexible
        output modes: per-token streaming or accumulated text, with or without
        thinking content.

        Args:
            token: Single token from LLM stream output.
            state: Mutable state dictionary containing:
                - response_text: Accumulated response text
                - is_in_thinking: Whether currently inside thinking tags
            show_thinking: If True, output thinking tags content; if False, replace or skip.
            return_per_token: If True, yield individual tokens; if False, accumulate and return full text.
            out_token_in_thinking_mode: Replacement token for thinking content when show_thinking=False.

        Returns:
            Processed token or accumulated text, or None if token should be skipped.

        Note:
            Thinking mode tokens are only filtered when show_thinking=False and
            out_token_in_thinking_mode is provided as a placeholder.
        """
        if show_thinking and return_per_token:
            return token
        elif show_thinking and not return_per_token:
            state['response_text'] += token
            return state['response_text']
        elif not show_thinking:
            if token in cls.opening_thinking_tags:
                state['is_in_thinking'] = True
                token = out_token_in_thinking_mode
                if token:
                    return token
            elif token in cls.closing_thinking_tags:
                state['is_in_thinking'] = False
                return
            if not state['is_in_thinking']:
                if return_per_token:
                    return token
                elif not return_per_token:
                    state['response_text'] += token
                    return state['response_text']

    @classmethod
    def prepare_gradio_chatbot_messages_to_openai(
        cls,
        system_prompt: str,
        support_system_role: bool,
        history_len: int,
        user_message: str,
        image_path: str | Path,
        resize_size: int | None,
        chatbot: GRADIO_CHAT_HISTORY,
        convert_to_openai_format: bool = True,
        use_responses_api: bool = False,
    ) -> list[dict]:
        """
        Convert Gradio chatbot messages to OpenAI API format.
        
        Takes a Gradio chatbot conversation history and transforms it into
        the format expected by OpenAI-compatible APIs, with optional image
        processing and system prompt handling.
        
        Args:
            system_prompt: System instructions for the model
            support_system_role: Whether the API supports system role
            history_len: Number of previous exchanges to include
            user_message: Current user input text
            image_path: Optional image for multimodal query
            resize_size: Maximum dimension for image resizing
            chatbot: Gradio chatbot history in standard format
            convert_to_openai_format: Whether to convert image messages to OpenAI format
            use_responses_api: Determines which API format to use
            
        Returns:
            List of messages formatted for OpenAI API
        """
        messages = []
        if support_system_role and system_prompt:
            messages.append(dict(role='system', content=system_prompt))
        if history_len != 0:
            messages.extend(chatbot[:-1][-(history_len*2):])
            if convert_to_openai_format:
                messages = cls._prepare_gradio_chatbot_image_messages_to_openai(
                    messages=messages,
                    resize_size=resize_size,
                    use_responses_api=use_responses_api,
                )
        text_message = cls._create_message_from_text(
            text=user_message,
            use_responses_api=use_responses_api,
        )
        if not image_path:
            messages.append(text_message)
            return messages
        message = cls._create_message_from_image(
            text=user_message,
            image_path=image_path,
            resize_size=resize_size,
            use_responses_api=use_responses_api,
        )
        if message:
            messages.append(message)
        else:
            messages.append(text_message)
        return messages

    @classmethod
    def _prepare_gradio_chatbot_image_messages_to_openai(
        cls,
        messages: GRADIO_CHAT_HISTORY,
        resize_size: int,
        use_responses_api: bool,
    ) -> GRADIO_CHAT_HISTORY:
        """
        Convert image content in Gradio messages to OpenAI format.
        
        Processes Gradio's multimodal message format (with 'file' type) and
        converts image content to base64 format expected by OpenAI APIs.
        
        Args:
            messages: Gradio chatbot messages with potential image content
            resize_size: Maximum dimension for image resizing
            use_responses_api: Determines which API format to use
            
        Returns:
            Converted messages with images in OpenAI-compatible format
            
        Reference:
            https://www.gradio.app/guides/creating-a-chatbot-fast#multimodal-chat-interface
        """
        openai_messages = []
        messages
        for i in range(len(messages)):
            curr_message = messages[i]
            openai_message = dict(role=curr_message['role'], content=[])
            debug_logger.debug(f"curr_message content: {curr_message['content']}")
            if not isinstance(curr_message['content'], list):
                continue
            for content in curr_message['content']:
                if content.get('type') == 'text':
                    text_content = cls._create_message_content_from_text(
                        text=content['text'],
                        use_responses_api=use_responses_api,
                    )
                    openai_message['content'].append(text_content)
                elif content.get('type') == 'file':
                    path = content.get('file', {}).get('path')
                    if Path(path).suffix.lower() in cls.image_extension:
                        image_content = cls._create_message_content_from_image(
                            image_path=path,
                            resize_size=resize_size,
                            use_responses_api=use_responses_api,
                        )
                        openai_message['content'].append(image_content)
            if openai_message['content']:
                openai_messages.append(openai_message)
        return openai_messages
