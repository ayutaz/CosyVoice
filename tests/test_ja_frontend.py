import re

import pytest

from cosyvoice.utils.ja_frontend import contains_japanese, should_ja_normalize, ja_text_to_katakana, ja_normalize

KANJI_OR_HIRAGANA_OR_DIGIT = re.compile(r'[一-鿿ぁ-ゟ0-9０-９]')


def test_contains_japanese():
    assert contains_japanese('こんにちは') is True
    assert contains_japanese('漢字とカナの文') is True
    # chinese / english must not be detected as japanese
    assert contains_japanese('收到好友从远方寄来的生日礼物') is False
    assert contains_japanese('hello world') is False


def test_should_ja_normalize_skips_control_tokens():
    assert should_ja_normalize('これは本です。') is True
    # control tokens do not survive g2p, such text must stay untouched
    assert should_ja_normalize('[breath]コレ ワ ホン デス。[breath]') is False
    assert should_ja_normalize('ここを[j][ǐ]と読む。') is False
    assert should_ja_normalize('hello world') is False


def test_official_example_sentence():
    # expected output taken from example.py cosyvoice3_example japanese usage note
    text = '歴史的世界においては、過去は単に過ぎ去ったものではない、プラトンのいう如く非有が有である。'
    expected = 'レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。'
    assert ja_text_to_katakana(text) == expected


def test_particle_pronunciation():
    assert ja_text_to_katakana('これは本です。') == 'コレ ワ ホン デス。'


def test_number_and_counter_reading():
    out = ja_text_to_katakana('総裁選挙は9月12日に告示し、27日に投開票を行うことを決めました。')
    assert 'クガツ' in out
    assert KANJI_OR_HIRAGANA_OR_DIGIT.search(out) is None


@pytest.mark.parametrize('text', [
    '自民党は、岸田総理大臣の後任を選ぶ総裁選挙について、9月12日に告示することを決めました。',
    '晴子、お前が見つけてきた変な男は、湘北に必要な男になったぞ',
    'YouTubeでAI動画を見た。',
    '2024年の売上は1,234億円でした。',
])
def test_output_is_katakana_only(text):
    out = ja_text_to_katakana(text)
    assert KANJI_OR_HIRAGANA_OR_DIGIT.search(out) is None
    assert len(out) > 0


def test_ja_normalize_split():
    sentence = '歴史的世界においては、過去は単に過ぎ去ったものではない。'
    texts = ja_normalize(sentence * 10)
    assert len(texts) >= 2
    for seg in texts:
        assert len(seg) <= 120
        assert KANJI_OR_HIRAGANA_OR_DIGIT.search(seg) is None
    # no content is lost by splitting
    assert ''.join(texts) == ja_text_to_katakana((sentence * 10).strip())


def test_ja_normalize_no_split():
    out = ja_normalize('これは本です。', split=False)
    assert out == 'コレ ワ ホン デス。'


def test_ja_normalize_empty_and_punctuation_only():
    assert ja_normalize('') == []
    assert ja_normalize('、、。') == []
