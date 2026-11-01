"""Smoke test for the ncnn-vulkan image.

Run inside a built image (CI does this, and it works locally too):

    docker run --rm --entrypoint python3 \
        -v "$PWD/docker/ncnn-vulkan/smoke_test.py:/smoke_test.py:ro" \
        -w /opt/frigate frigate-ncnn-vulkan:latest /smoke_test.py

Each check guards against a failure that has actually shipped or nearly
shipped:

* ``pip install ncnn`` pulling NumPy 2, which breaks onnxruntime (built
  against NumPy 1.x) so Frigate cannot start.
* ``pip install ncnn`` pulling plain ``opencv-python`` over
  ``opencv-contrib-python``, which removes ``cv2.face`` (face landmarks).
* The Rust libraries failing to load (missing file, bad glibc symbol
  versions, wrong exports), in which case Frigate silently falls back to
  the slower Python paths and nothing else notices.
"""

import ctypes
import os
import subprocess
import sys
import unittest

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, then exit
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        return
    if result is False:
        FAILURES.append(name)
        print(f"FAIL  {name}")
    else:
        print(f"ok    {name}")


def numpy_is_1x() -> bool:
    import numpy

    print(f"      numpy {numpy.__version__}")
    return numpy.__version__.startswith("1.")


def onnxruntime_imports() -> None:
    import onnxruntime  # noqa: F401


def opencv_contrib_present() -> bool:
    import cv2

    return hasattr(cv2, "face") and hasattr(cv2.face, "createFacemarkLBF")


def ncnn_imports() -> None:
    import ncnn  # noqa: F401


def libncnn_loads() -> None:
    ctypes.CDLL("/opt/frigate/libncnn.so")


def frame_rs_loads() -> bool:
    from frigate.util.frame_rs import frame_rs_available

    return frame_rs_available()


def motion_rs_loads() -> bool:
    from frigate.motion.rust_engine import motion_available

    return motion_available()


def yolo_rs_loads() -> bool:
    from frigate.detectors.rust_yolo import yolo_available

    return yolo_available()


def detector_binary_runs() -> bool:
    # With no arguments the binary prints its usage line and exits 1. Getting
    # that far proves it is executable and its glibc symbols resolve.
    proc = subprocess.run(
        ["/opt/frigate/frigate-detector-rs"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode == 1 and "Usage" in proc.stderr


def frigate_app_imports() -> None:
    import frigate.app  # noqa: F401


INTEGRATION_TESTS = [
    "frigate.test.test_video",
    "frigate.test.test_stationary_classifier",
    "frigate.test.test_face_preview_yolo_fixes",
    "frigate.test.test_stress_concurrency",
]


def main() -> int:
    # Keep our lines in order with unittest's stderr output in CI logs.
    sys.stdout.reconfigure(line_buffering=True)

    if os.path.isdir("/opt/frigate"):
        os.chdir("/opt/frigate")
        sys.path.insert(0, "/opt/frigate")

    check("numpy is 1.x (onnxruntime ABI)", numpy_is_1x)
    check("onnxruntime imports", onnxruntime_imports)
    check("opencv-contrib present (cv2.face.createFacemarkLBF)", opencv_contrib_present)
    check("ncnn python module imports", ncnn_imports)
    check("libncnn.so loads", libncnn_loads)
    check("libfrigate_frame_rs.so loads", frame_rs_loads)
    check("libfrigate_motion_rs.so loads", motion_rs_loads)
    check("libfrigate_yolo_rs.so loads", yolo_rs_loads)
    check("frigate-detector-rs executes", detector_binary_runs)
    check("frigate.app imports", frigate_app_imports)

    print("\nRust-path integration tests:")
    suite = unittest.defaultTestLoader.loadTestsFromNames(INTEGRATION_TESTS)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        FAILURES.append("integration tests")

    if FAILURES:
        print("\nSmoke test FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
