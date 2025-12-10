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

"""Tests for cosyvoice/utils/frontend_utils.py"""

import pytest
import inflect

from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    replace_blank,
    is_only_punctuation,
)


class TestContainsChinese:
    """Tests for contains_chinese function."""

    def test_chinese_only(self):
        assert contains_chinese('你好世界') is True

    def test_english_only(self):
        assert contains_chinese('Hello World') is False

    def test_mixed_chinese_english(self):
        assert contains_chinese('Hello 你好') is True

    def test_empty_string(self):
        assert contains_chinese('') is False

    def test_numbers_only(self):
        assert contains_chinese('12345') is False

    def test_japanese_hiragana(self):
        # Hiragana is not in Chinese character range
        assert contains_chinese('こんにちは') is False

    def test_chinese_punctuation(self):
        # Chinese punctuation without Chinese characters
        assert contains_chinese('，。！') is False


class TestReplaceCornerMark:
    """Tests for replace_corner_mark function."""

    def test_square_symbol(self):
        assert replace_corner_mark('10²') == '10平方'

    def test_cube_symbol(self):
        assert replace_corner_mark('5³') == '5立方'

    def test_both_symbols(self):
        assert replace_corner_mark('面积10²体积5³') == '面积10平方体积5立方'

    def test_no_symbols(self):
        assert replace_corner_mark('normal text') == 'normal text'

    def test_empty_string(self):
        assert replace_corner_mark('') == ''


class TestRemoveBracket:
    """Tests for remove_bracket function."""

    def test_chinese_parentheses(self):
        assert remove_bracket('你好（世界）') == '你好世界'

    def test_chinese_brackets(self):
        assert remove_bracket('你好【世界】') == '你好世界'

    def test_backticks(self):
        assert remove_bracket('`code`') == 'code'

    def test_em_dash(self):
        assert remove_bracket('hello——world') == 'hello world'

    def test_mixed_brackets(self):
        assert remove_bracket('（test）【example】') == 'testexample'

    def test_empty_string(self):
        assert remove_bracket('') == ''


class TestSpellOutNumber:
    """Tests for spell_out_number function."""

    @pytest.fixture
    def inflect_parser(self):
        return inflect.engine()

    def test_single_digit(self, inflect_parser):
        result = spell_out_number('5', inflect_parser)
        assert result == 'five'

    def test_multiple_digits(self, inflect_parser):
        result = spell_out_number('123', inflect_parser)
        assert result == 'one hundred and twenty-three'

    def test_number_in_text(self, inflect_parser):
        result = spell_out_number('I have 3 apples', inflect_parser)
        assert 'three' in result
        assert 'I have' in result
        assert 'apples' in result

    def test_no_numbers(self, inflect_parser):
        result = spell_out_number('hello world', inflect_parser)
        assert result == 'hello world'

    def test_multiple_numbers(self, inflect_parser):
        result = spell_out_number('1 and 2', inflect_parser)
        assert 'one' in result
        assert 'two' in result

    def test_empty_string(self, inflect_parser):
        result = spell_out_number('', inflect_parser)
        assert result == ''


class TestReplaceBlank:
    """Tests for replace_blank function."""

    def test_blank_between_ascii(self):
        # Space between ASCII characters should be preserved
        result = replace_blank('a b')
        assert result == 'a b'

    def test_blank_between_chinese(self):
        # Space between Chinese characters should be removed
        result = replace_blank('你 好')
        assert result == '你好'

    def test_blank_between_mixed(self):
        # Space between Chinese and ASCII should be removed
        result = replace_blank('你 a')
        assert result == '你a'

    def test_no_blanks(self):
        result = replace_blank('hello')
        assert result == 'hello'


class TestIsOnlyPunctuation:
    """Tests for is_only_punctuation function."""

    def test_punctuation_only(self):
        assert is_only_punctuation('...') is True
        assert is_only_punctuation('，。！') is True
        assert is_only_punctuation('?!.') is True

    def test_empty_string(self):
        assert is_only_punctuation('') is True

    def test_text_with_punctuation(self):
        assert is_only_punctuation('hello!') is False

    def test_text_only(self):
        assert is_only_punctuation('hello') is False

    def test_numbers(self):
        assert is_only_punctuation('123') is False

    def test_symbols(self):
        # Symbols like currency should match
        assert is_only_punctuation('$') is True
        assert is_only_punctuation('¥') is True
