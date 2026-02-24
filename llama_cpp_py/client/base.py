import re
import io
import base64
from pathlib import Path

from PIL import Image

from llama_cpp_py.logger import debug_logger


class LlamaBaseClient:
    """
    Base client for interacting with LLM models, providing common preprocessing
    and postprocessing utilities for text generation.
    
    Handles thinking tags removal
    """
    opening_thinking_tags = ['<think>', '&lt;think&gt;']
    closing_thinking_tags = ['</think>', '&lt;/think&gt;']
    all_thinking_tags = [*opening_thinking_tags, *closing_thinking_tags]
    image_extension = ['.png', '.jpg', '.jpeg', '.webp']

    @classmethod
    def _prepare_messages(
        cls,
        user_message_or_messages: str | list[dict],
        system_prompt: str,
        image_path_or_base64: str | Path,
        resize_size: int | None,
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
            image_path_or_base64: Path to image file or base64 encoded string.
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
        if not image_path_or_base64:
            messages.append(dict(role='user', content=user_message_or_messages))
            return messages
        image_message = cls._create_completion_message_from_image(
            text=user_message_or_messages,
            image_path_or_base64=image_path_or_base64,
            resize_size=resize_size,
        )
        if image_message:
            messages.append(image_message)
        return messages


    @classmethod
    def _prepare_image(cls, image: str | Path, resize_size: int | None) -> str | None:
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
        if isinstance(image, (str, Path)):
            image_path = Path(image)
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


    @classmethod
    def _create_completion_message_from_image(
        cls,
        image_path_or_base64: str | Path,
        resize_size: int | None,
        text: str = '',
    ) -> dict:
        image_base64 = cls._prepare_image(
            image=image_path_or_base64,
            resize_size=resize_size,
        )
        if image_base64:
            message = dict(role='user', content=[
                dict(type='image_url', image_url=dict(url=f'data:image/png;base64,{image_base64}')),
                dict(type='text', text=text),
            ])
            return message


    @classmethod
    def _process_output_token(
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
