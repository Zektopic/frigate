"""Tests for the ONNX Runtime session options chosen for each model."""

import unittest
from unittest.mock import patch

import onnxruntime as ort

from frigate.detectors import detection_runners
from frigate.detectors.detection_runners import (
    get_optimized_runner,
    get_ort_session_options,
    is_cpu_enrichment_session,
)
from frigate.embeddings.types import EnrichmentModelTypeEnum

SPINNING = "session.intra_op.allow_spinning"


def spinning_entry(options: ort.SessionOptions) -> str | None:
    try:
        return options.get_session_config_entry(SPINNING)
    except RuntimeError:
        return None


class TestGetOrtSessionOptions(unittest.TestCase):
    def test_defaults_need_no_options(self):
        self.assertIsNone(get_ort_session_options())
        self.assertIsNone(get_ort_session_options(False, disable_spinning=False))

    def test_disable_spinning_sets_the_entry(self):
        options = get_ort_session_options(disable_spinning=True)

        self.assertEqual(spinning_entry(options), "0")
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

    def test_complex_model_keeps_spinning(self):
        options = get_ort_session_options(True)

        self.assertIsNone(spinning_entry(options))
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        )

    def test_complex_model_on_cpu_gets_both(self):
        options = get_ort_session_options(True, disable_spinning=True)

        self.assertEqual(spinning_entry(options), "0")
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        )


class TestIsCpuEnrichmentSession(unittest.TestCase):
    def test_every_enrichment_model_on_cpu(self):
        for model_type in EnrichmentModelTypeEnum:
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    is_cpu_enrichment_session(
                        model_type.value, ["CPUExecutionProvider"]
                    )
                )

    def test_gpu_sessions_keep_spinning(self):
        self.assertFalse(
            is_cpu_enrichment_session(
                EnrichmentModelTypeEnum.arcface.value,
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        )

    def test_detector_models_keep_spinning(self):
        # detectors run continuously, where spinning threads lower latency
        for model_type in [None, "yolo-generic", "rfdetr"]:
            with self.subTest(model_type=model_type):
                self.assertFalse(
                    is_cpu_enrichment_session(model_type, ["CPUExecutionProvider"])
                )

    def test_no_providers(self):
        self.assertFalse(
            is_cpu_enrichment_session(EnrichmentModelTypeEnum.arcface.value, [])
        )


class TestOptimizedRunnerSessionOptions(unittest.TestCase):
    def test_only_cpu_enrichment_sessions_disable_spinning(self):
        for provider, model_type, expected in [
            ("CPUExecutionProvider", "paddleocr", "0"),
            ("CUDAExecutionProvider", "paddleocr", None),
            ("CPUExecutionProvider", "yolo-generic", None),
        ]:
            with (
                self.subTest(provider=provider, model_type=model_type),
                patch.object(
                    detection_runners, "is_rknn_compatible", return_value=False
                ),
                patch.object(
                    detection_runners,
                    "get_ort_providers",
                    return_value=([provider], [{"device_id": 0}]),
                ),
                patch.object(
                    detection_runners,
                    "is_openvino_gpu_npu_available",
                    return_value=False,
                ),
                patch.object(detection_runners.ort, "InferenceSession") as session,
            ):
                session.return_value.get_providers.return_value = [provider]

                # paddleocr skips the CUDA graph runner, which takes no options
                get_optimized_runner("/models/model.onnx", "GPU", model_type)

                options = session.call_args.kwargs["sess_options"]
                spinning = None if options is None else spinning_entry(options)

                self.assertEqual(spinning, expected)


if __name__ == "__main__":
    unittest.main()
