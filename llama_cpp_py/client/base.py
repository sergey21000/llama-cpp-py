import re
from pathlib import Path


class LlamaBaseClient:
    """
    Base client for interacting with LLM models, providing common preprocessing
    and postprocessing utilities for text generation.
    
    Handles thinking tags removal, transliteration, and text cleaning for TTS.
    """
    opening_thinking_tags: list[str] = ['<think>', '&lt;think&gt;']
    closing_thinking_tags: list[str] = ['</think>', '&lt;/think&gt;']
    all_thinking_tags = [*opening_thinking_tags, *closing_thinking_tags]

    @classmethod
    def clean_thinking_tags(cls, text: str) -> str:
        """
        Remove thinking tags (like <think>...</think>) from the text.
        
        Args:
            text: Input text potentially containing thinking tags.
            
        Returns:
            Text with thinking tags and their content removed.
        """
        for open_tag, close_tag in zip(cls.opening_thinking_tags, cls.closing_thinking_tags):
            pattern = rf'{open_tag}.*?{close_tag}'
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        return text

    @staticmethod
    def transliterate_english_to_russian(text: str) -> str:
        """
        Convert English characters and symbols to Russian phonetic equivalents.
        
        Useful for TTS systems that handle Russian text better.
        
        Args:
            text: Text containing English characters, numbers, and symbols.
            
        Returns:
            Text with transliterated characters.
        """
        translit_map = {
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е',
            'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и', 'j': 'ж',
            'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
            'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т',
            'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й',
            'z': 'з',
            'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е',
            'F': 'Ф', 'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Ж',
            'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О',
            'P': 'П', 'Q': 'К', 'R': 'Р', 'S': 'С', 'T': 'Т',
            'U': 'У', 'V': 'В', 'W': 'В', 'X': 'КС', 'Y': 'Й',
            'Z': 'З',
            '0': 'ноль', '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
            '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь', '9': 'девять',
            '+': ' плюс ', '-': ' минус ', '=': ' равно ',
            '*': ' умножить ', '/': ' разделить ', '%': ' процент ',
        }

        def transliterate_char(char):
            return translit_map.get(char, char)

        return ''.join(transliterate_char(c) for c in text)


    @classmethod
    def clean_text_before_speech(cls, text: str) -> str:
        """
        Prepare text for TTS by removing unwanted characters and normalizing.
        
        Performs:
        1. Thinking tags removal
        2. English-to-Russian transliteration
        3. Special character filtering
        4. Whitespace normalization
        
        Args:
            text: Raw text from LLM output.
            
        Returns:
            Cleaned text suitable for speech synthesis.
        """
        text = cls.clean_thinking_tags(text)
        text = cls.transliterate_english_to_russian(text)
        text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,!?;:()\"'-]", '', text)
        text = re.sub(r'[\"\'«»]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    @classmethod
    def _prepare_messages(
        cls,
        user_message_or_messages: str,
        system_prompt: str,
        image_path_or_base64: str | Path = '',
        resize_size: int = 512,
        support_system_role: bool = True,
    ) -> list[dict[str, str]]:
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
        image_base64 = cls._prepare_image(
            image=image_path_or_base64,
            resize_size=resize_size,
        )
        messages.append(dict(role='user', content=[
            dict(type='image_url', image_url=dict(url=f'data:image/png;base64,{image_base64}')),
            dict(type='text', text=user_message_or_messages),
        ]))
        return messages

    @staticmethod
    def _prepare_image(image: str | Path, resize_size: int) -> str:
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
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                image_pil = Image.open(image_path).convert('RGB')
                image_pil.thumbnail((resize_size, resize_size))
                buffer = io.BytesIO()
                image_pil.save(buffer, format='PNG')
                image = base64.b64encode(buffer.getvalue()).decode()
        return image

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
