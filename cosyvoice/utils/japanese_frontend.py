# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

"""Japanese text processing support for CosyVoice"""

import re
from typing import List, Tuple

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False
    print("Warning: pyopenjtalk not found. Japanese text normalization will be limited.")


class JapaneseTextNormalizer:
    """Japanese text normalization class using pyopenjtalk"""

    def __init__(self, use_phoneme: bool = False):
        """
        Args:
            use_phoneme: If True, convert text to phoneme sequence
        """
        self.use_phoneme = use_phoneme

        # Full-width to half-width conversion table
        self.zenkaku_to_hankaku = str.maketrans(
            'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
            '０１２３４５６７８９',
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
        )

    def normalize(self, text: str) -> str:
        """Normalize Japanese text

        Args:
            text: Input text (can contain kanji, hiragana, katakana, alphabet)

        Returns:
            Normalized text
        """
        if not text:
            return text

        # 1. Convert full-width alphanumerics to half-width
        text = text.translate(self.zenkaku_to_hankaku)

        # 2. Normalize special characters
        text = self._normalize_special_chars(text)

        # 3. Remove consecutive whitespaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _normalize_special_chars(self, text: str) -> str:
        """Normalize special characters"""
        replacements = {
            '　': ' ',      # Full-width space
            '！': '!',
            '？': '?',
            '（': '(',
            '）': ')',
            '「': '',       # Remove Japanese quotes (preserve for ADV games if needed)
            '」': '',
            '『': '',
            '』': '',
            '【': '',
            '】': '',
            '―': '-',
            '〜': '~',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def text_to_phoneme(self, text: str) -> str:
        """Convert text to phoneme sequence (for inference)

        Args:
            text: Input text

        Returns:
            Phoneme sequence (space-separated)
        """
        if not PYOPENJTALK_AVAILABLE:
            raise RuntimeError("pyopenjtalk is required for phoneme conversion")

        phonemes = pyopenjtalk.g2p(text, kana=False)
        return phonemes

    def text_to_kana(self, text: str) -> str:
        """Convert text to katakana reading

        Args:
            text: Input text

        Returns:
            Katakana reading
        """
        if not PYOPENJTALK_AVAILABLE:
            raise RuntimeError("pyopenjtalk is required for kana conversion")

        kana = pyopenjtalk.g2p(text, kana=True)
        return kana

    def get_accent_info(self, text: str) -> List[Tuple[str, int]]:
        """Get accent information from text

        Args:
            text: Input text

        Returns:
            List of (reading, accent_type) tuples
        """
        if not PYOPENJTALK_AVAILABLE:
            raise RuntimeError("pyopenjtalk is required for accent info")

        njd_features = pyopenjtalk.run_frontend(text)
        result = []
        for feature in njd_features:
            string = feature['string']
            acc = feature.get('acc', 0)
            result.append((string, acc))
        return result


def normalize_japanese_text(text: str) -> str:
    """Utility function to normalize Japanese text"""
    normalizer = JapaneseTextNormalizer()
    return normalizer.normalize(text)
