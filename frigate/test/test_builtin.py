import unittest
from unittest.mock import MagicMock
import sys

# Mocking numpy and ruamel.yaml as they are missing in the current test environment.
# These mocks are used to allow the tests to run without installing these heavy dependencies.
# Note: process_config_query_string is a pure function and doesn't depend on these.
sys.modules['numpy'] = MagicMock()
sys.modules['ruamel'] = MagicMock()
sys.modules['ruamel.yaml'] = MagicMock()

from frigate.util.builtin import process_config_query_string

class TestBuiltin(unittest.TestCase):
    def test_process_config_query_string_multiple_values(self):
        query_string = {'key': ['val1', 'val2']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': ['val1', 'val2']})

    def test_process_config_query_string_single_int(self):
        query_string = {'key': ['123']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': 123})

    def test_process_config_query_string_single_float(self):
        query_string = {'key': ['123.45']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': 123.45})

    def test_process_config_query_string_single_bool_true(self):
        query_string = {'key': ['True']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': True})

    def test_process_config_query_string_single_bool_false(self):
        query_string = {'key': ['False']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': False})

    def test_process_config_query_string_single_list_literal(self):
        query_string = {'key': ["['a', 'b']"]}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': ['a', 'b']})

    def test_process_config_query_string_single_dict_literal(self):
        query_string = {'key': ["{'a': 1}"]}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': {'a': 1}})

    def test_process_config_query_string_single_dict_literal_with_comma(self):
        query_string = {'key': ["{'a': 1, 'b': 2}"]}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': {'a': 1, 'b': 2}})

    def test_process_config_query_string_single_string_with_comma(self):
        # Strings with commas should be treated as literal strings (masks/zones)
        query_string = {'key': ['0,0,1,1,2,2']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': '0,0,1,1,2,2'})

    def test_process_config_query_string_single_plain_string(self):
        query_string = {'key': ['hello']}
        result = process_config_query_string(query_string)
        self.assertEqual(result, {'key': 'hello'})

    def test_process_config_query_string_mixed(self):
        query_string = {
            'int': ['1'],
            'list': ['val1', 'val2'],
            'mask': ['0,0,1,1'],
            'bool': ['True'],
            'complex_list': ["['a', 'b', 'c']"],
            'complex_dict': ["{'x': 1, 'y': 2}"]
        }
        expected = {
            'int': 1,
            'list': ['val1', 'val2'],
            'mask': '0,0,1,1',
            'bool': True,
            'complex_list': ['a', 'b', 'c'],
            'complex_dict': {'x': 1, 'y': 2}
        }
        result = process_config_query_string(query_string)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
