import unittest
from unittest.mock import patch, MagicMock
import types

# Create a mock for 'lib' package structure
lib_mock = types.ModuleType('lib')
lib_mock.__path__ = []

# Mock modules using a dictionary to apply them safely in tests
MOCK_MODULES = {
    'yaml': MagicMock(),
    'lib': lib_mock,
    'lib.i18n_loader': MagicMock(),
    'lib.ui_generator': MagicMock(),
    'lib.yaml_extractor': MagicMock(),
    'lib.nav_map': MagicMock(),
    'lib.schema_loader': MagicMock(),
    'lib.file_utils': MagicMock(),
    'lib.section_config_parser': MagicMock(),
}

class TestNormalizeWhitespace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We must mock dependencies before importing the module to test.
        # Use patch.dict to avoid global module pollution.
        cls.patcher = patch.dict('sys.modules', MOCK_MODULES)
        cls.patcher.start()

        # Import the function after patching
        global _normalize_whitespace
        from docs.scripts.generate_ui_tabs import _normalize_whitespace

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_empty_string(self):
        """Test with empty string."""
        self.assertEqual(_normalize_whitespace(""), "")

    def test_single_line(self):
        """Test with simple single line."""
        self.assertEqual(_normalize_whitespace("hello world"), "hello world")

    def test_trailing_whitespace(self):
        """Test that trailing whitespace is removed."""
        self.assertEqual(_normalize_whitespace("hello \t\n"), "hello")

    def test_leading_whitespace_overall(self):
        """Test that leading whitespace on the entire string is removed due to global strip()."""
        self.assertEqual(_normalize_whitespace("  hello"), "hello")

    def test_leading_whitespace_on_inner_lines_is_preserved(self):
        """Test that inner lines preserve their leading whitespace (only right-stripped)."""
        input_str = "hello\n  world"
        self.assertEqual(_normalize_whitespace(input_str), "hello\n  world")

    def test_multiple_blank_lines(self):
        """Test collapsing multiple blank lines into one."""
        self.assertEqual(
            _normalize_whitespace("line1\n\n\nline2"),
            "line1\n\nline2"
        )

    def test_multiple_blank_lines_with_whitespace(self):
        """Test that lines containing only whitespace are treated as blank lines."""
        self.assertEqual(
            _normalize_whitespace("line1\n \n  \nline2"),
            "line1\n\nline2"
        )

    def test_complex_whitespace(self):
        """Test a combination of leading, trailing, blank lines and tabs."""
        # Note: the entire string gets .strip() so first '\n  ' and trailing '\t\n' are removed.
        # The inner blank line with \t gets .rstrip() and becomes a blank line.
        input_str = "\n  line1  \n\n \t \n  line2\t\n"
        expected = "line1\n\n  line2"
        self.assertEqual(_normalize_whitespace(input_str), expected)

    def test_string_literals(self):
        """Test that string literals (quotes) inside the text are not altered."""
        input_str = 'key: "value with spaces"\nother_key: \'value with spaces\''
        self.assertEqual(_normalize_whitespace(input_str), input_str)

if __name__ == '__main__':
    unittest.main()
