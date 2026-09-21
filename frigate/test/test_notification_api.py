import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules.setdefault("py_vapid", MagicMock())
sys.modules.setdefault("py3nvml", MagicMock())
sys.modules.setdefault("py3nvml.py3nvml", MagicMock())
sys.modules.setdefault("zmq", MagicMock())
sys.modules.setdefault("frigate.version", MagicMock())

from frigate.api.notification import register_notifications


class TestNotificationApi(unittest.TestCase):
    def test_register_notifications_missing_sub(self):
        req = MagicMock()
        res = register_notifications(req, body={})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Subscription must be provided", res.body)

    def test_register_notifications_anonymous_rejected(self):
        req = MagicMock()
        req.headers = {"remote-user": "anonymous"}
        res = register_notifications(req, body={"sub": "token123"})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Cannot register notifications for an anonymous user", res.body)

    def test_register_notifications_no_user_header_rejected(self):
        req = MagicMock()
        req.headers = {}
        res = register_notifications(req, body={"sub": "token123"})
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Cannot register notifications for an anonymous user", res.body)

    @patch("frigate.api.notification.User")
    def test_register_notifications_user_not_found(self, mock_user):
        req = MagicMock()
        req.headers = {"remote-user": "nonexistent"}
        mock_user.update.return_value.where.return_value.execute.return_value = 0
        res = register_notifications(req, body={"sub": "token123"})
        self.assertEqual(res.status_code, 404)
        self.assertIn(b"Could not find user", res.body)

    @patch("frigate.api.notification.User")
    def test_register_notifications_success(self, mock_user):
        req = MagicMock()
        req.headers = {"remote-user": "test_user"}
        mock_user.update.return_value.where.return_value.execute.return_value = 1
        res = register_notifications(req, body={"sub": "token123"})
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Successfully saved token", res.body)


if __name__ == "__main__":
    unittest.main()
