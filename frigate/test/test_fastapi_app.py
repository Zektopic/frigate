import unittest
from unittest.mock import MagicMock

from frigate.api.fastapi_app import check_csrf


class TestCheckCSRF(unittest.TestCase):
    def test_safe_methods_allowed(self):
        """Test that safe HTTP methods are allowed even without CSRF token."""
        for method in ["GET", "HEAD", "OPTIONS", "TRACE"]:
            request = MagicMock()
            request.method = method
            request.headers = {}
            self.assertTrue(check_csrf(request))

    def test_unsafe_methods_without_token_rejected(self):
        """Test that unsafe HTTP methods without CSRF token are rejected."""
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            request = MagicMock()
            request.method = method
            request.headers = {}
            self.assertFalse(check_csrf(request))

    def test_unsafe_methods_with_token_allowed(self):
        """Test that unsafe HTTP methods with CSRF token are allowed."""
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            request = MagicMock()
            request.method = method
            request.headers = {"x-csrf-token": "some-token"}
            self.assertTrue(check_csrf(request))
