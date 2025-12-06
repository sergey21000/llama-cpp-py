import re



class LlamaClientBase:
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


    @staticmethod
    def _prepare_messages(
        user_message_or_messages: str,
        system_prompt: str,
        support_system_role: bool = True,
    ) -> list[dict[str, str]]:
        """
        Format messages for LLM API call.
        
        Args:
            user_message_or_messages: Either a single user message string or
                                      a list of pre-formatted message dicts.
            system_prompt: System instructions for the model.
            support_system_role: Whether to include system role in messages.
            
        Returns:
            List of message dictionaries in OpenAI format.
        """
        if isinstance(user_message_or_messages, list):
            return user_message_or_messages
        messages = []
        if support_system_role and system_prompt:
            messages.append(dict(role='system', content=system_prompt))
        messages.append(dict(role='user', content=user_message_or_messages))
        return messages


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
        Process individual token from LLM stream with thinking mode handling.
        
        Args:
            token: Single token from LLM output.
            state: Mutable state dict tracking response text and thinking mode.
            show_thinking: Whether to output thinking tags content.
            return_per_token: If True, yield individual tokens; if False, accumulate.
            out_token_in_thinking_mode: Replacement token for thinking mode content.
            
        Returns:
            Processed token or accumulated text based on flags, or None if skipped.
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
