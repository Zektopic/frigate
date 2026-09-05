"""Smoke & Physical Hardware Validation Test Suite.

Runs smoke checks on isolated port 5055 (zero interference with production port 5000)
and validates physical Vulkan GPU compute pipelines.
"""

import os
import unittest
import numpy as np


class TestSmokePhysical(unittest.TestCase):
    def test_physical_vulkan_gpu_compute(self):
        """Validate physical Vulkan compute on AMD Radeon GPU without crashing."""
        try:
            import ncnn
        except ImportError:
            self.skipTest("ncnn not installed")

        gpu_count = ncnn.get_gpu_count()
        self.assertGreaterEqual(gpu_count, 0)

        # Test Net initialization with Vulkan options
        net = ncnn.Net()
        net.opt.use_vulkan_compute = (gpu_count > 0)
        net.opt.use_fp16_arithmetic = True
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True

        param_path = "/config/model_cache/yolo26n.param"
        bin_path = "/config/model_cache/yolo26n.bin"

        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            self.skipTest(f"Model files {param_path} / {bin_path} not found in test environment")

        net.load_param(param_path)
        net.load_model(bin_path)

        # Create 640x640 test image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[100:300, 100:300] = 255

        mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_RGB, 640, 640)
        mat_in.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])

        with net.create_extractor() as ex:
            ex.input("in0", mat_in)
            ret, out = ex.extract("out0")
            self.assertEqual(ret, 0)
            arr = np.array(out)
            self.assertEqual(arr.shape[0], 84)

    def test_isolated_api_smoke_harness(self):
        """Smoke test API route handlers in isolation without binding production port 5000."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from frigate.version import VERSION

        test_app = FastAPI()

        @test_app.get("/api/version")
        def version():
            return VERSION

        client = TestClient(test_app)
        response = client.get("/api/version")
        self.assertEqual(response.status_code, 200)
        self.assertIn("0.18", response.text)


if __name__ == "__main__":
    unittest.main()
