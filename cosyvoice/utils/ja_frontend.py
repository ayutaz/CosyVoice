# Copyright (c) 2026 Alibaba Inc
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
"""Japanese text frontend.

CosyVoice models share kanji tokens with Chinese, so raw kanji-mixed Japanese
text is frequently read with Chinese readings (see example.py, which requires
Japanese input as space-separated katakana). This module converts kanji-mixed
Japanese into that expected format using pyopenjtalk (OpenJTalk NJD), which
also resolves number/counter readings and particle pronunciations (は -> ワ).
"""
import re

from cosyvoice.utils.frontend_utils import split_paragraph, is_only_punctuation

# hiragana / katakana / prolonged sound mark
_KANA_PATTERN = re.compile(r'[ぁ-ゟァ-ヺー]')

_PUNC_KEEP = {'、', '。', '！', '？'}
_PUNC_MAP = {
    ',': '、', '，': '、', ':': '、', '：': '、', '‥': '、',
    '.': '。', '．': '。', ';': '。', '；': '。', '…': '。',
    '!': '！', '?': '？',
}


_CONTROL_TOKEN_PATTERN = re.compile(r'\[[^\]]*\]')


def contains_japanese(text):
    # NOTE kanji-only text cannot be distinguished from Chinese and is not detected
    return bool(_KANA_PATTERN.search(text))


def should_ja_normalize(text):
    # NOTE inline control tokens ([breath], [laughter], pinyin hotfix etc.) do not survive g2p,
    # such text must be provided as katakana manually
    return contains_japanese(text) and not _CONTROL_TOKEN_PATTERN.search(text)


def ja_text_to_katakana(text):
    try:
        import pyopenjtalk
    except ImportError:
        raise ImportError('pyopenjtalk is required for japanese text normalization, install with `pip install pyopenjtalk`')
    pieces = []
    for feat in pyopenjtalk.run_frontend(text):
        surface, pron, pos = feat['string'], feat['pron'], feat['pos']
        if pos == '記号':
            # NOTE njd rewrites 。 pron to 、, keep the surface form instead and drop unknown symbols
            punc = surface if surface in _PUNC_KEEP else _PUNC_MAP.get(surface, '')
            if punc and pieces:
                pieces.append(punc)
            continue
        # NOTE drop accent marks, they are not part of the katakana input format
        pron = pron.replace('’', '')
        if pos == '助詞' and pron == 'ヲ':
            pron = 'オ'
        if pron in ('', '*'):
            continue
        pieces.append(pron)
    text = ' '.join(pieces)
    # punctuation attaches to the previous word, no space on either side
    text = re.sub(r' ?([、。！？]) ?', r'\1', text)
    return text


def ja_normalize(text, split=True, token_max_n=80, token_min_n=60, merge_len=20):
    text = ja_text_to_katakana(text.strip())
    if split is False:
        return text
    if text == '':
        return []
    # NOTE length in "zh" mode is character based, no tokenizer is needed
    texts = list(split_paragraph(text, None, "zh", token_max_n=token_max_n,
                                 token_min_n=token_min_n, merge_len=merge_len, comma_split=False))
    return [i for i in texts if not is_only_punctuation(i)]
