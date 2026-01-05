# Copyright (c) 2024 CosyVoice Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Japanese text preprocessing utilities for CosyVoice3."""

import re


def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese-specific characters (hiragana or katakana).

    This function specifically checks for hiragana and katakana, which are
    unique to Japanese. Kanji (CJK ideographs) are not checked because they
    are shared with Chinese.

    Args:
        text: Input text to check.

    Returns:
        True if text contains hiragana or katakana.
    """
    # U+3040-U+309F: Hiragana (Japanese-specific)
    # U+30A0-U+30FF: Katakana (Japanese-specific)
    # Note: Kanji (U+4E00-U+9FFF) is shared with Chinese, so not included
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text))


def katakana_to_hiragana(text: str) -> str:
    """Convert katakana to hiragana.

    Args:
        text: Input text with katakana.

    Returns:
        Text with katakana converted to hiragana.
    """
    result = []
    for c in text:
        # Katakana range: U+30A1 (ァ) to U+30F6 (ヶ)
        # Hiragana range: U+3041 (ぁ) to U+3096 (ゖ)
        # Offset: 0x60
        if '\u30A1' <= c <= '\u30F6':
            result.append(chr(ord(c) - 0x60))
        else:
            result.append(c)
    return ''.join(result)


def add_word_spaces(text: str) -> str:
    """Add spaces around punctuation marks.

    Args:
        text: Input text.

    Returns:
        Text with spaces around punctuation.
    """
    # Add spaces around Japanese punctuation
    text = re.sub(r'([、。！？])', r' \1 ', text)
    # Remove multiple consecutive spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def preprocess_japanese(text: str) -> str:
    """Preprocess Japanese text for TTS.

    This function:
    1. Preserves language tags like <|ja|>
    2. Segments text into words using sudachipy
    3. Converts kanji to hiragana reading using pyopenjtalk
    4. Joins words with spaces

    Args:
        text: Input Japanese text.

    Returns:
        Preprocessed text suitable for CosyVoice3 inference.
    """
    import pyopenjtalk
    from sudachipy import Dictionary, SplitMode

    # Preserve language tag if present
    lang_tag = ""
    if text.startswith("<|ja|>"):
        lang_tag = "<|ja|>"
        text = text[6:]

    try:
        # Word segmentation using sudachipy
        tokenizer = Dictionary().create()
        tokens = tokenizer.tokenize(text, SplitMode.C)  # SplitMode.C for finest granularity

        # Convert each word to hiragana reading
        words = []
        for token in tokens:
            surface = token.surface()
            # Get reading (katakana) from sudachi, fallback to pyopenjtalk
            reading = token.reading_form()
            if reading:
                # Convert katakana to hiragana
                reading = katakana_to_hiragana(reading)
            else:
                # Fallback: use pyopenjtalk for kanji conversion
                try:
                    reading = pyopenjtalk.g2p(surface, kana=True)
                    reading = katakana_to_hiragana(reading)
                except Exception:
                    reading = surface
            words.append(reading)

        processed = ' '.join(words)
    except Exception:
        # Fallback: use pyopenjtalk only
        try:
            processed = pyopenjtalk.g2p(text, kana=True)
            processed = katakana_to_hiragana(processed)
            processed = add_word_spaces(processed)
        except Exception:
            return lang_tag + text

    return lang_tag + processed
