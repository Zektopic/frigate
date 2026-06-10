import unittest
from unittest.mock import MagicMock, patch

try:
    from frigate.util.services import get_fs_type
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    get_fs_type = None  # type: ignore[assignment]

@unittest.skipIf(not SERVICES_AVAILABLE, "OpenCV not installed — required by frigate.util.services")
class TestGetFsType(unittest.TestCase):
    def setUp(self):
        # Create mock disk partitions
        self.mock_partitions = [
            MagicMock(mountpoint="/", fstype="ext4"),
            MagicMock(mountpoint="/mnt/data", fstype="btrfs"),
            MagicMock(mountpoint="/mnt/data2", fstype="xfs"),
            MagicMock(mountpoint="/var/log", fstype="ext3"),
        ]

    @patch("frigate.util.services.psutil.disk_partitions")
    def test_get_fs_type_exact_match(self, mock_disk_partitions):
        mock_disk_partitions.return_value = self.mock_partitions
        self.assertEqual(get_fs_type("/"), "ext4")
        self.assertEqual(get_fs_type("/mnt/data"), "btrfs")

    @patch("frigate.util.services.psutil.disk_partitions")
    def test_get_fs_type_subpath_match(self, mock_disk_partitions):
        mock_disk_partitions.return_value = self.mock_partitions
        self.assertEqual(get_fs_type("/home/user/file.txt"), "ext4")
        self.assertEqual(get_fs_type("/mnt/data/movies/movie.mp4"), "btrfs")
        self.assertEqual(get_fs_type("/mnt/data2/music/song.mp3"), "xfs")
        self.assertEqual(get_fs_type("/var/log/frigate.log"), "ext3")

    @patch("frigate.util.services.psutil.disk_partitions")
    def test_get_fs_type_longest_match(self, mock_disk_partitions):
        # Add a more specific mountpoint
        self.mock_partitions.append(
            MagicMock(mountpoint="/mnt/data/special", fstype="tmpfs")
        )
        mock_disk_partitions.return_value = self.mock_partitions

        # Should match the longest prefix
        self.assertEqual(get_fs_type("/mnt/data/special/file.txt"), "tmpfs")
        self.assertEqual(get_fs_type("/mnt/data/other/file.txt"), "btrfs")

    @patch("frigate.util.services.psutil.disk_partitions")
    def test_get_fs_type_partial_match_false_positive(self, mock_disk_partitions):
        # "/mnt/data_old" should not match "/mnt/data"
        mock_disk_partitions.return_value = self.mock_partitions

        # With the fix, this should now correctly fall back to the root mount "/"
        self.assertEqual(get_fs_type("/mnt/data_old/file.txt"), "ext4")

if __name__ == "__main__":
    unittest.main()
