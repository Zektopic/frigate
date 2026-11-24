"""main() must raise the open file limit before it starts any process."""

import unittest
from unittest.mock import patch

import frigate.__main__ as frigate_main


class _Stop(Exception):
    pass


class TestMainRaisesFileLimitFirst(unittest.TestCase):
    def test_file_limit_is_raised_before_the_forkserver_starts(self) -> None:
        # mp.Manager() is the first thing that starts a process, and with the
        # forkserver start method every later Frigate process is forked from
        # the server it starts, inheriting its open file limit.
        calls: list[str] = []

        def start_manager():
            calls.append("manager")
            raise _Stop

        with (
            patch.object(
                frigate_main,
                "set_file_limit",
                side_effect=lambda: calls.append("file_limit"),
            ),
            patch.object(frigate_main.mp, "Manager", side_effect=start_manager),
            self.assertRaises(_Stop),
        ):
            frigate_main.main()

        self.assertEqual(calls, ["file_limit", "manager"])


if __name__ == "__main__":
    unittest.main()
