"""Test camera user and password cleanup."""

import unittest

from frigate.util.builtin import clean_camera_user_pass, escape_special_characters


class TestUserPassCleanup(unittest.TestCase):
    def setUp(self) -> None:
        self.rtsp_with_pass = "rtsp://user:password@192.168.0.2:554/live"
        self.rtsp_with_special_pass = (
            "rtsp://user:password`~!@#$%^&*()-_;',.<>:\"\{\}\[\]@@192.168.0.2:554/live"
        )
        self.rtsp_no_pass = "rtsp://192.168.0.3:554/live"

    def test_cleanup(self):
        """Test that user / pass are cleaned up."""
        clean = clean_camera_user_pass(self.rtsp_with_pass)
        assert clean != self.rtsp_with_pass
        assert "user:password" not in clean

    def test_no_cleanup(self):
        """Test that nothing changes when no user / pass are defined."""
        clean = clean_camera_user_pass(self.rtsp_no_pass)
        assert clean == self.rtsp_no_pass

    def test_special_char_password(self):
        """Test that special characters in pw are escaped, but not others."""
        escaped = escape_special_characters(self.rtsp_with_special_pass)
        assert (
            escaped
            == "rtsp://user:password%60~%21%40%23%24%25%5E%26%2A%28%29-_%3B%27%2C.%3C%3E%3A%22%5C%7B%5C%7D%5C%5B%5C%5D%40@192.168.0.2:554/live"
        )

    def test_no_special_char_password(self):
        """Test that no change is made to path with no special characters."""
        escaped = escape_special_characters(self.rtsp_with_pass)
        assert escaped == self.rtsp_with_pass


class TestUserPassMasking(unittest.TestCase):
    def setUp(self) -> None:
        self.rtsp_log_message = "Did you mean file:rtsp://user:password@192.168.1.3:554"

    def test_rtsp_in_log_message(self):
        """Test that the rtsp url in a log message is escaped."""
        escaped = clean_camera_user_pass(self.rtsp_log_message)
        print(f"The escaped is {escaped}")
        assert escaped == "Did you mean file:rtsp://*:*@192.168.1.3:554"


class TestUserPassMaskingUsernames(unittest.TestCase):
    """Credentials must be masked whatever characters the username uses."""

    def test_dotted_username_is_masked(self):
        clean = clean_camera_user_pass("rtsp://john.doe:s3cret@192.168.1.3:554/live")
        assert "s3cret" not in clean
        assert clean == "rtsp://*:*@192.168.1.3:554/live"

    def test_percent_encoded_username_is_masked(self):
        clean = clean_camera_user_pass("rtsp://admin%40site:s3cret@cam.local/live")
        assert "s3cret" not in clean

    def test_empty_username_is_masked(self):
        clean = clean_camera_user_pass("rtsp://:s3cret@192.168.1.3/live")
        assert "s3cret" not in clean

    def test_http_query_dotted_username_is_masked(self):
        clean = clean_camera_user_pass(
            "http://192.168.1.3/cgi-bin/api.cgi?cmd=Snap&user=john.doe&password=s3cret"
        )
        assert "s3cret" not in clean
        assert "user=*&password=*" in clean

    def test_host_with_port_is_untouched(self):
        url = "rtsp://192.168.1.3:554/live"
        assert clean_camera_user_pass(url) == url
