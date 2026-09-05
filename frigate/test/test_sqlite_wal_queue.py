import os
import tempfile
import unittest
from frigate.db.sqlitevecq import SqliteVecQueueDatabase


class TestSqliteWalQueue(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_frigate.db")
        self.db = SqliteVecQueueDatabase(self.db_path)

    def tearDown(self):
        self.db.stop()
        self.tmp_dir.cleanup()

    def test_wal_pragmas_applied(self):
        """Verify that WAL mode and busy_timeout are active."""
        conn = self.db._connect()
        cursor = conn.cursor()

        # Check journal_mode
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        self.assertEqual(journal_mode.lower(), "wal")

        # Check busy_timeout
        cursor.execute("PRAGMA busy_timeout;")
        timeout = cursor.fetchone()[0]
        self.assertEqual(timeout, 30000)

        # Check synchronous
        cursor.execute("PRAGMA synchronous;")
        sync = cursor.fetchone()[0]
        # NORMAL mode is 1
        self.assertEqual(sync, 1)


if __name__ == "__main__":
    unittest.main()
