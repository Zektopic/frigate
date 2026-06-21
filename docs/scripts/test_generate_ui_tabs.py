import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add docs/scripts to path so we can import generate_ui_tabs
sys.path.insert(0, str(Path(__file__).resolve().parent))

class MockModule(MagicMock):
    """Custom mock module that prevents unittest.TestLoader errors"""
    @property
    def __path__(self): raise AttributeError
    @property
    def __file__(self): raise AttributeError
    @property
    def __loader__(self): raise AttributeError
    @property
    def __spec__(self): raise AttributeError
    @property
    def __name__(self): raise AttributeError
    @property
    def __mro_entries__(self): raise AttributeError
    @property
    def __origin__(self): raise AttributeError
    @property
    def __args__(self): raise AttributeError
    @property
    def __parameters__(self): raise AttributeError

# We need to mock these dependencies because generate_ui_tabs.py
# attempts to import them at the module level.
mock_yaml = MockModule()
mock_pydantic = MockModule()
mock_frigate = MockModule()

with patch.dict('sys.modules', {
    'yaml': mock_yaml,
    'pydantic': mock_pydantic,
    'pydantic.schema': mock_pydantic,
    'frigate': mock_frigate,
}):
    # Now we can safely import the module
    from generate_ui_tabs import _ensure_imports

class TestEnsureImports(unittest.TestCase):
    def test_no_components_no_imports_added(self):
        content = "---\nid: test\n---\n\n# Hello\nWorld!"
        result = _ensure_imports(content)
        self.assertEqual(content, result)

    def test_components_present_imports_added_after_frontmatter(self):
        content = "---\nid: test\n---\n\n# Hello\n<ConfigTabs>\n<TabItem>\n<NavPath>"
        expected = '---\nid: test\n---\n\nimport ConfigTabs from "@site/src/components/ConfigTabs";\nimport TabItem from "@theme/TabItem";\nimport NavPath from "@site/src/components/NavPath";\n\n\n# Hello\n<ConfigTabs>\n<TabItem>\n<NavPath>'
        result = _ensure_imports(content)
        self.assertEqual(expected, result)

    def test_components_present_no_frontmatter(self):
        content = "# Hello\n<ConfigTabs>"
        expected = '\nimport ConfigTabs from "@site/src/components/ConfigTabs";\n\n# Hello\n<ConfigTabs>'
        result = _ensure_imports(content)
        self.assertEqual(expected, result)

    def test_some_imports_already_present(self):
        content = '---\nid: test\n---\nimport ConfigTabs from "@site/src/components/ConfigTabs";\n\n# Hello\n<ConfigTabs>\n<TabItem>'
        expected = '---\nid: test\n---\n\nimport TabItem from "@theme/TabItem";\n\nimport ConfigTabs from "@site/src/components/ConfigTabs";\n\n# Hello\n<ConfigTabs>\n<TabItem>'
        result = _ensure_imports(content)
        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
