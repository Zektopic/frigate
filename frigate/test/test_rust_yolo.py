import unittest
from unittest.mock import patch, MagicMock

import frigate.detectors.rust_yolo as rust_yolo
from frigate.detectors.rust_yolo import yolo_available

class TestRustYolo(unittest.TestCase):
    def setUp(self):
        # Reset the global state before each test
        rust_yolo._lib = None
        rust_yolo._available = None

    @patch('frigate.detectors.rust_yolo._load_lib')
    def test_yolo_available_true(self, mock_load_lib):
        mock_load_lib.return_value = MagicMock()
        self.assertTrue(yolo_available())
        mock_load_lib.assert_called_once()

    @patch('frigate.detectors.rust_yolo._load_lib')
    def test_yolo_available_false(self, mock_load_lib):
        mock_load_lib.return_value = None
        self.assertFalse(yolo_available())
        mock_load_lib.assert_called_once()

    @patch('os.path.isfile')
    @patch('ctypes.CDLL')
    def test_load_lib_success(self, mock_cdll, mock_isfile):
        # Test the actual _load_lib function
        mock_isfile.return_value = True
        mock_cdll.return_value = MagicMock()

        lib = rust_yolo._load_lib()

        self.assertIsNotNone(lib)
        self.assertTrue(rust_yolo._available)
        self.assertIsNotNone(rust_yolo._lib)

    @patch('os.path.isfile')
    @patch('ctypes.CDLL')
    def test_load_lib_oserror(self, mock_cdll, mock_isfile):
        mock_isfile.return_value = True
        mock_cdll.side_effect = OSError("Library not found")

        lib = rust_yolo._load_lib()

        self.assertIsNone(lib)
        self.assertFalse(rust_yolo._available)

    @patch('os.path.isfile')
    def test_load_lib_not_found(self, mock_isfile):
        mock_isfile.return_value = False

        lib = rust_yolo._load_lib()

        self.assertIsNone(lib)
        self.assertFalse(rust_yolo._available)
