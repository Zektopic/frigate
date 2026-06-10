import unittest

from frigate.api.auth import validate_password_strength


class TestValidatePasswordStrength(unittest.TestCase):
    def test_empty_password(self):
        """Test that an empty password fails validation."""
        is_valid, error = validate_password_strength("")
        self.assertFalse(is_valid)
        self.assertEqual(error, "Password cannot be empty")

    def test_none_password(self):
        """Test that a None password fails validation."""
        is_valid, error = validate_password_strength(None)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Password cannot be empty")

    def test_short_password(self):
        """Test that a password shorter than 12 characters fails validation."""
        is_valid, error = validate_password_strength("short")
        self.assertFalse(is_valid)
        self.assertEqual(error, "Password must be at least 12 characters long")

        is_valid, error = validate_password_strength("12345678901")
        self.assertFalse(is_valid)
        self.assertEqual(error, "Password must be at least 12 characters long")

    def test_exact_length_password(self):
        """Test that a password exactly 12 characters long passes validation."""
        is_valid, error = validate_password_strength("123456789012")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_long_password(self):
        """Test that a password longer than 12 characters passes validation."""
        is_valid, error = validate_password_strength("thisisalongpassword123!")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
