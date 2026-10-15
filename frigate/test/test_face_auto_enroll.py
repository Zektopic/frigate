"""Tests for guarded face auto-enrollment."""

import unittest

from frigate.data_processing.common.face.auto_enroll import (
    AUTO_TAG,
    AutoEnroller,
    AutoEnrollSettings,
)


class FakeImage:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.copied = False

    def copy(self) -> "FakeImage":
        clone = FakeImage(self.tag)
        clone.copied = True
        return clone


class FakeLibrary:
    def __init__(self, people: dict[str, list[str]]) -> None:
        self.files = {name: list(files) for name, files in people.items()}
        self.written: list[tuple[str, str, object]] = []

    def people(self) -> set[str]:
        return set(self.files)

    def list_files(self, name: str) -> list[str]:
        return list(self.files.get(name, []))

    def write(self, name: str, filename: str, image: object) -> None:
        self.files[name].append(filename)
        self.written.append((name, filename, image))

    def delete(self, name: str, filename: str) -> None:
        self.files[name].remove(filename)


SETTINGS = AutoEnrollSettings(
    enabled=True,
    min_score=0.95,
    min_margin=0.15,
    min_frames=3,
    min_area=2500,
    max_images=3,
    cooldown_seconds=900,
)


def feed(
    enroller: AutoEnroller,
    obj: str,
    name: str = "alice",
    n: int = 3,
    score: float = 0.97,
    runner_up: float = 0.5,
    area: int = 10000,
    t0: float = 1000.0,
) -> None:
    for i in range(n):
        enroller.observe(
            obj,
            name,
            score + i * 0.001,
            runner_up,
            area,
            FakeImage(f"{obj}-{i}"),
            t0 + i,
        )


class TestFaceAutoEnroll(unittest.TestCase):
    def setUp(self) -> None:
        self.lib = FakeLibrary({"alice": ["alice_1.webp"], "bob": []})
        self.enrolled: list[str] = []
        self.enroller = AutoEnroller(
            SETTINGS, self.lib, on_enrolled=self.enrolled.append
        )

    def test_enrolls_after_consistent_confident_frames(self):
        feed(self.enroller, "obj1")
        self.assertEqual(self.enroller.finish("obj1", now=2000), "alice")
        self.assertEqual(self.enrolled, ["alice"])
        name, filename, image = self.lib.written[0]
        self.assertEqual(name, "alice")
        self.assertIn(AUTO_TAG, filename)
        # best (highest score) frame is kept, and it is a private copy
        self.assertEqual(image.tag, "obj1-2")
        self.assertTrue(image.copied)

    def test_disabled_does_nothing(self):
        enroller = AutoEnroller(AutoEnrollSettings(enabled=False), self.lib)
        feed(enroller, "obj1")
        self.assertIsNone(enroller.finish("obj1", now=2000))
        self.assertEqual(self.lib.written, [])

    def test_too_few_frames(self):
        feed(self.enroller, "obj1", n=2)
        self.assertIsNone(self.enroller.finish("obj1", now=2000))

    def test_low_score_or_margin_or_area_do_not_qualify(self):
        feed(self.enroller, "a", score=0.90)
        feed(self.enroller, "b", runner_up=0.9)
        feed(self.enroller, "c", area=100)
        for obj in "abc":
            self.assertIsNone(self.enroller.finish(obj, now=2000))
        self.assertEqual(self.lib.written, [])

    def test_conflicting_names_block_enrollment(self):
        feed(self.enroller, "obj1")
        # a single weak recognition as someone else is enough to veto
        self.enroller.observe("obj1", "bob", 0.85, 0.8, 10000, FakeImage("x"), 1010)
        self.assertIsNone(self.enroller.finish("obj1", now=2000))

    def test_unknown_frames_do_not_veto(self):
        feed(self.enroller, "obj1")
        self.enroller.observe("obj1", "unknown", 0.3, 0.2, 10000, FakeImage("x"), 1010)
        self.assertEqual(self.enroller.finish("obj1", now=2000), "alice")

    def test_never_creates_new_people(self):
        feed(self.enroller, "obj1", name="mallory")
        self.assertIsNone(self.enroller.finish("obj1", now=2000))
        self.assertNotIn("mallory", self.lib.files)

    def test_cooldown_per_person(self):
        feed(self.enroller, "obj1")
        self.assertEqual(self.enroller.finish("obj1", now=2000), "alice")
        feed(self.enroller, "obj2")
        self.assertIsNone(self.enroller.finish("obj2", now=2000 + 899))
        feed(self.enroller, "obj3")
        self.assertEqual(self.enroller.finish("obj3", now=2000 + 900), "alice")

    def test_rotation_keeps_manual_images(self):
        settings = AutoEnrollSettings(**{**SETTINGS.__dict__, "cooldown_seconds": 0})
        enroller = AutoEnroller(settings, self.lib)
        for i in range(5):
            feed(enroller, f"obj{i}", t0=1000.0 + i * 100)
            enroller.finish(f"obj{i}", now=5000 + i)
        files = self.lib.files["alice"]
        autos = sorted(f for f in files if AUTO_TAG in f)
        self.assertIn("alice_1.webp", files)  # manual image untouched
        self.assertEqual(len(autos), SETTINGS.max_images)
        # the newest auto images survive
        self.assertTrue(autos[-1].startswith(f"alice{AUTO_TAG}1402"))

    def test_finish_unknown_object_is_noop(self):
        self.assertIsNone(self.enroller.finish("never-seen", now=1))


if __name__ == "__main__":
    unittest.main()
