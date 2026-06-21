import unittest
import sys
import os
import importlib.util

class TestExtractTranslationsFromSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need to test the extract_translations_from_schema function, but it's
        # in a script that runs top-level logic when imported. To avoid importing
        # all of Frigate's missing dependencies in the test environment, we will
        # parse the script using AST and compile it dynamically, avoiding the
        # module-level imports.

        import ast

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(root_dir, "generate_config_translations.py")

        with open(script_path, "r") as f:
            tree = ast.parse(f.read())

        # Filter out import statements and top-level calls to isolate function definitions
        filtered_body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'extract_translations_from_schema':
                filtered_body.append(node)

        # Create a new module AST containing only our target function
        new_tree = ast.Module(body=filtered_body, type_ignores=[])
        ast.fix_missing_locations(new_tree)

        # Compile and execute in a temporary namespace
        compiled = compile(new_tree, filename="<ast>", mode="exec")
        cls.namespace = {}
        # Provide typing dependency since it uses Dict/Any
        from typing import Dict, Any
        cls.namespace.update({'Dict': Dict, 'Any': Any})
        exec(compiled, cls.namespace)

    def setUp(self):
        self.extract_translations_from_schema = self.__class__.namespace['extract_translations_from_schema']

    def test_simple_schema(self):
        schema = {
            "title": "Camera Config",
            "description": "Configuration for a camera.",
            "properties": {
                "name": {
                    "title": "Camera Name",
                    "description": "Name of the camera."
                }
            }
        }
        expected = {
            "label": "Camera Config",
            "description": "Configuration for a camera.",
            "name": {
                "label": "Camera Name",
                "description": "Name of the camera."
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

    def test_nested_schema(self):
        schema = {
            "title": "Config",
            "properties": {
                "mqtt": {
                    "title": "MQTT Config",
                    "properties": {
                        "host": {
                            "title": "Host",
                            "description": "MQTT Host"
                        }
                    }
                }
            }
        }
        expected = {
            "label": "Config",
            "mqtt": {
                "label": "MQTT Config",
                "host": {
                    "label": "Host",
                    "description": "MQTT Host"
                }
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

    def test_schema_with_defs_and_ref(self):
        schema = {
            "$defs": {
                "MqttConfig": {
                    "title": "MQTT Config Default",
                    "properties": {
                        "port": {
                            "title": "Port",
                        }
                    }
                }
            },
            "properties": {
                "mqtt": {
                    "$ref": "#/$defs/MqttConfig",
                    "title": "MQTT Configuration Override"
                }
            }
        }
        expected = {
            "mqtt": {
                "label": "MQTT Configuration Override",
                "port": {
                    "label": "Port"
                }
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

    def test_schema_with_anyOf(self):
        schema = {
            "title": "Camera Config",
            "properties": {
                "motion": {
                    "title": "Motion Config",
                    "anyOf": [
                        {"type": "null"},
                        {
                            "properties": {
                                "threshold": {
                                    "title": "Threshold",
                                    "description": "Motion threshold"
                                }
                            }
                        }
                    ]
                }
            }
        }
        expected = {
            "label": "Camera Config",
            "motion": {
                "label": "Motion Config",
                "threshold": {
                    "label": "Threshold",
                    "description": "Motion threshold"
                }
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

    def test_schema_with_items_and_ref(self):
        schema = {
            "$defs": {
                "ZoneConfig": {
                    "title": "Zone Config",
                    "properties": {
                        "coordinates": {
                            "title": "Coordinates"
                        }
                    }
                }
            },
            "properties": {
                "zones": {
                    "title": "Zones List",
                    "items": {
                        "$ref": "#/$defs/ZoneConfig"
                    }
                }
            }
        }
        expected = {
            "zones": {
                "label": "Zones List",
                "coordinates": {
                    "label": "Coordinates"
                }
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

    def test_schema_with_additionalProperties_and_ref(self):
        schema = {
            "$defs": {
                "CameraConfig": {
                    "title": "Camera Configuration",
                    "properties": {
                        "name": {
                            "title": "Name"
                        }
                    }
                }
            },
            "properties": {
                "cameras": {
                    "title": "Cameras Map",
                    "additionalProperties": {
                        "$ref": "#/$defs/CameraConfig"
                    }
                }
            }
        }
        expected = {
            "cameras": {
                "label": "Cameras Map",
                "name": {
                    "label": "Name"
                }
            }
        }
        self.assertEqual(self.extract_translations_from_schema(schema), expected)

if __name__ == '__main__':
    unittest.main()
