# Copyright (c) 2024 Alibaba Inc
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

"""
Japanese text processing utilities for CosyVoice

This module provides utilities to convert Japanese text (with kanji) to hiragana
using pyopenjtalk, resolving reading ambiguities for better TTS pronunciation.
"""

import logging

try:
    import pyopenjtalk
    PYOPENJTALK_AVAILABLE = True
except ImportError:
    PYOPENJTALK_AVAILABLE = False
    logging.warning("pyopenjtalk-plus not installed, Japanese processing will be unavailable")


def convert_phonemes_to_hiragana(phonemes):
    """
    Convert pyopenjtalk phonemes to hiragana

    Args:
        phonemes (str): Space-separated phonemes from pyopenjtalk.g2p()
                       e.g., "ky o o w a y o i t e N k i d e s U"

    Returns:
        str: Hiragana representation
             e.g., "きょうはよいてんきです"
    """
    # Phoneme to hiragana mapping
    phoneme_map = {
        # Vowels
        'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
        # K-row
        'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
        'kya': 'きゃ', 'kyu': 'きゅ', 'kyo': 'きょ',
        # G-row
        'ga': 'が', 'gi': 'ぎ', 'gu': 'ぐ', 'ge': 'げ', 'go': 'ご',
        'gya': 'ぎゃ', 'gyu': 'ぎゅ', 'gyo': 'ぎょ',
        # S-row
        'sa': 'さ', 'si': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
        'sha': 'しゃ', 'shu': 'しゅ', 'sho': 'しょ', 'sh': 'し',
        'sya': 'しゃ', 'syu': 'しゅ', 'syo': 'しょ',
        # Z-row
        'za': 'ざ', 'zi': 'じ', 'zu': 'ず', 'ze': 'ぜ', 'zo': 'ぞ',
        'ja': 'じゃ', 'ju': 'じゅ', 'jo': 'じょ', 'j': 'じ',
        'zya': 'じゃ', 'zyu': 'じゅ', 'zyo': 'じょ',
        # T-row
        'ta': 'た', 'ti': 'ち', 'tu': 'つ', 'te': 'て', 'to': 'と',
        'cha': 'ちゃ', 'chu': 'ちゅ', 'cho': 'ちょ', 'ch': 'ち',
        'tya': 'ちゃ', 'tyu': 'ちゅ', 'tyo': 'ちょ',
        'tsu': 'つ', 'ts': 'つ',
        # D-row
        'da': 'だ', 'di': 'ぢ', 'du': 'づ', 'de': 'で', 'do': 'ど',
        'dya': 'ぢゃ', 'dyu': 'ぢゅ', 'dyo': 'ぢょ',
        # N-row
        'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
        'nya': 'にゃ', 'nyu': 'にゅ', 'nyo': 'にょ',
        # H-row
        'ha': 'は', 'hi': 'ひ', 'hu': 'ふ', 'he': 'へ', 'ho': 'ほ',
        'hya': 'ひゃ', 'hyu': 'ひゅ', 'hyo': 'ひょ',
        # B-row
        'ba': 'ば', 'bi': 'び', 'bu': 'ぶ', 'be': 'べ', 'bo': 'ぼ',
        'bya': 'びゃ', 'byu': 'びゅ', 'byo': 'びょ',
        # P-row
        'pa': 'ぱ', 'pi': 'ぴ', 'pu': 'ぷ', 'pe': 'ぺ', 'po': 'ぽ',
        'pya': 'ぴゃ', 'pyu': 'ぴゅ', 'pyo': 'ぴょ',
        # M-row
        'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
        'mya': 'みゃ', 'myu': 'みゅ', 'myo': 'みょ',
        # Y-row
        'ya': 'や', 'yi': 'い', 'yu': 'ゆ', 'ye': 'いぇ', 'yo': 'よ',
        # R-row
        'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
        'rya': 'りゃ', 'ryu': 'りゅ', 'ryo': 'りょ',
        # W-row
        'wa': 'わ', 'wi': 'ゐ', 'wu': 'う', 'we': 'ゑ', 'wo': 'を', 'w': 'わ',
        # N
        'n': 'ん', 'N': 'ん',
        # Partial consonants
        'ky': 'き', 'gy': 'ぎ', 'sy': 'し', 'zy': 'じ',
        'ty': 'ち', 'dy': 'ぢ', 'ny': 'に', 'hy': 'ひ',
        'by': 'び', 'py': 'ぴ', 'my': 'み', 'ry': 'り',
        'k': 'く', 'g': 'ぐ', 's': 'す', 'z': 'ず',
        't': 'と', 'd': 'ど', 'h': 'ふ', 'b': 'ぶ',
        'p': 'ぷ', 'm': 'む', 'y': 'ゆ', 'r': 'る',
        # V-row (foreign sounds)
        'v': 'ゔ', 'va': 'ゔぁ', 'vi': 'ゔぃ', 'vu': 'ゔ', 've': 'ゔぇ', 'vo': 'ゔぉ',
        # Devoiced vowels (capital letters in pyopenjtalk)
        'A': 'あ', 'I': 'い', 'U': 'う', 'E': 'え', 'O': 'お',
        # Special
        'pau': '、', 'sil': '', 'cl': 'っ',
    }

    # Convert phoneme string to hiragana
    phoneme_list = phonemes.split()
    hiragana_list = []
    i = 0

    while i < len(phoneme_list):
        matched = False

        # Try to match longest phoneme sequence first (3, then 2, then 1)
        for lookahead in [2, 1, 0]:
            if i + lookahead < len(phoneme_list):
                if lookahead == 2:
                    three_phoneme = phoneme_list[i] + phoneme_list[i+1] + phoneme_list[i+2]
                    if three_phoneme in phoneme_map:
                        hiragana_list.append(phoneme_map[three_phoneme])
                        i += 3
                        matched = True
                        break
                elif lookahead == 1:
                    two_phoneme = phoneme_list[i] + phoneme_list[i+1]
                    if two_phoneme in phoneme_map:
                        hiragana_list.append(phoneme_map[two_phoneme])
                        i += 2
                        matched = True
                        break
                else:  # lookahead == 0
                    phoneme = phoneme_list[i]
                    if phoneme in phoneme_map:
                        hiragana_list.append(phoneme_map[phoneme])
                        i += 1
                        matched = True
                        break

        if not matched:
            # Unknown phoneme - log and skip
            phoneme = phoneme_list[i]
            logging.debug(f'Unknown phoneme: {phoneme}')
            i += 1

    return ''.join(hiragana_list)


def convert_japanese_to_hiragana(text):
    """
    Convert Japanese text (with kanji) to hiragana using pyopenjtalk

    This resolves reading ambiguities in Japanese text by using pyopenjtalk's
    morphological analysis and phoneme conversion.

    Args:
        text (str): Japanese text with kanji
                   e.g., "今日は良い天気です。"

    Returns:
        str: Hiragana representation
             e.g., "きょうはよいてんきです"

    Raises:
        RuntimeError: If pyopenjtalk is not available
    """
    if not PYOPENJTALK_AVAILABLE:
        raise RuntimeError(
            "pyopenjtalk-plus is not installed. "
            "Install it with: pip install pyopenjtalk-plus"
        )

    # Get phonemes from pyopenjtalk
    phonemes = pyopenjtalk.g2p(text)

    # Convert phonemes to hiragana
    hiragana = convert_phonemes_to_hiragana(phonemes)

    logging.debug(f"Japanese conversion: '{text}' -> phonemes: '{phonemes}' -> hiragana: '{hiragana}'")

    return hiragana


def prepare_japanese_text_for_tts(text, add_language_tag=True):
    """
    Prepare Japanese text for TTS by converting kanji to hiragana
    and optionally adding language tag

    Args:
        text (str): Japanese text with kanji
        add_language_tag (bool): Whether to add <|jp|> tag

    Returns:
        str: Prepared text ready for TTS
             e.g., "<|jp|>きょうはよいてんきです"
    """
    hiragana = convert_japanese_to_hiragana(text)

    if add_language_tag:
        return f"<|jp|>{hiragana}"
    else:
        return hiragana
