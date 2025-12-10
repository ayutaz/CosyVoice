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

"""Tests for cosyvoice/utils/file_utils.py"""

import json
import os

import pytest

# file_utils.py imports torch at module level, skip if not available
pytest.importorskip('torch', reason='torch required for file_utils')

from cosyvoice.utils.file_utils import read_lists, read_json_lists


class TestReadLists:
    """Tests for read_lists function."""

    def test_read_simple_list(self, sample_list_file):
        result = read_lists(sample_list_file)
        assert result == ['line1', 'line2', 'line3']

    def test_read_empty_file(self, temp_dir):
        empty_file = os.path.join(temp_dir, 'empty.list')
        with open(empty_file, 'w', encoding='utf8') as f:
            pass
        result = read_lists(empty_file)
        assert result == []

    def test_read_file_with_whitespace(self, temp_dir):
        whitespace_file = os.path.join(temp_dir, 'whitespace.list')
        with open(whitespace_file, 'w', encoding='utf8') as f:
            f.write('  line1  \n')
            f.write('line2\n')
        result = read_lists(whitespace_file)
        assert result == ['line1', 'line2']

    def test_read_unicode_content(self, temp_dir):
        unicode_file = os.path.join(temp_dir, 'unicode.list')
        with open(unicode_file, 'w', encoding='utf8') as f:
            f.write('你好\n')
            f.write('世界\n')
        result = read_lists(unicode_file)
        assert result == ['你好', '世界']


class TestReadJsonLists:
    """Tests for read_json_lists function."""

    def test_read_single_json(self, temp_dir):
        # Create JSON file
        json_file = os.path.join(temp_dir, 'data.json')
        with open(json_file, 'w', encoding='utf8') as f:
            json.dump({'key1': 'value1', 'key2': 'value2'}, f)

        # Create list file pointing to JSON
        list_file = os.path.join(temp_dir, 'data.list')
        with open(list_file, 'w', encoding='utf8') as f:
            f.write(json_file + '\n')

        result = read_json_lists(list_file)
        assert result == {'key1': 'value1', 'key2': 'value2'}

    def test_read_multiple_json(self, temp_dir):
        # Create first JSON file
        json_file1 = os.path.join(temp_dir, 'data1.json')
        with open(json_file1, 'w', encoding='utf8') as f:
            json.dump({'key1': 'value1'}, f)

        # Create second JSON file
        json_file2 = os.path.join(temp_dir, 'data2.json')
        with open(json_file2, 'w', encoding='utf8') as f:
            json.dump({'key2': 'value2'}, f)

        # Create list file pointing to both JSONs
        list_file = os.path.join(temp_dir, 'data.list')
        with open(list_file, 'w', encoding='utf8') as f:
            f.write(json_file1 + '\n')
            f.write(json_file2 + '\n')

        result = read_json_lists(list_file)
        assert result == {'key1': 'value1', 'key2': 'value2'}

    def test_read_empty_list(self, temp_dir):
        list_file = os.path.join(temp_dir, 'empty.list')
        with open(list_file, 'w', encoding='utf8') as f:
            pass

        result = read_json_lists(list_file)
        assert result == {}
