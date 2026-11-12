import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pydantic import parse_obj_as

import frigate.detectors as detectors
import frigate.object_detection.base
from frigate.config import DetectorConfig, ModelConfig
from frigate.detectors import DetectorTypeEnum
from frigate.detectors.detector_config import InputDTypeEnum, InputTensorEnum
from frigate.object_detection.util import tensor_transform


class TestLocalObjectDetector(unittest.TestCase):
    def test_localdetectorprocess_should_only_create_specified_detector_type(self):
        for det_type in detectors.api_types:
            with self.subTest(det_type=det_type):
                with patch.dict(
                    "frigate.detectors.api_types",
                    {det_type: Mock() for det_type in DetectorTypeEnum},
                ):
                    test_cfg = parse_obj_as(
                        DetectorConfig, ({"type": det_type, "model": {}})
                    )
                    test_cfg.model.path = "/test/modelpath"
                    test_obj = frigate.object_detection.base.LocalObjectDetector(
                        detector_config=test_cfg
                    )

                    assert test_obj is not None
                    for api_key, mock_detector in detectors.api_types.items():
                        if test_cfg.type == api_key:
                            mock_detector.assert_called_once_with(test_cfg)
                        else:
                            mock_detector.assert_not_called()

    @patch.dict(
        "frigate.detectors.api_types",
        {det_type: Mock() for det_type in DetectorTypeEnum},
    )
    def test_detect_raw_given_tensor_input_should_return_api_detect_raw_result(self):
        mock_cputfl = detectors.api_types[DetectorTypeEnum.cpu]

        TEST_DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        TEST_DETECT_RESULT = np.ndarray([1, 2, 4, 8, 16, 32])
        test_obj_detect = frigate.object_detection.base.LocalObjectDetector(
            detector_config=parse_obj_as(DetectorConfig, {"type": "cpu", "model": {}})
        )

        mock_det_api = mock_cputfl.return_value
        mock_det_api.detect_raw.return_value = TEST_DETECT_RESULT

        test_result = test_obj_detect.detect_raw(TEST_DATA)

        mock_det_api.detect_raw.assert_called_once_with(tensor_input=TEST_DATA)
        assert test_result is mock_det_api.detect_raw.return_value

    @patch.dict(
        "frigate.detectors.api_types",
        {det_type: Mock() for det_type in DetectorTypeEnum},
    )
    def test_detect_raw_given_tensor_input_should_call_api_detect_raw_with_transposed_tensor(
        self,
    ):
        mock_cputfl = detectors.api_types[DetectorTypeEnum.cpu]

        TEST_DATA = np.zeros((1, 32, 32, 3), np.uint8)
        TEST_DETECT_RESULT = np.ndarray([1, 2, 4, 8, 16, 32])

        test_cfg = parse_obj_as(DetectorConfig, {"type": "cpu", "model": {}})
        test_cfg.model.input_tensor = InputTensorEnum.nchw

        test_obj_detect = frigate.object_detection.base.LocalObjectDetector(
            detector_config=test_cfg
        )

        mock_det_api = mock_cputfl.return_value
        mock_det_api.detect_raw.return_value = TEST_DETECT_RESULT

        test_result = test_obj_detect.detect_raw(TEST_DATA)

        mock_det_api.detect_raw.assert_called_once()
        assert (
            mock_det_api.detect_raw.call_args.kwargs["tensor_input"].shape
            == np.zeros((1, 3, 32, 32)).shape
        )

        assert test_result is mock_det_api.detect_raw.return_value

    @patch.dict(
        "frigate.detectors.api_types",
        {det_type: Mock() for det_type in DetectorTypeEnum},
    )
    @patch("frigate.object_detection.base.load_labels")
    def test_detect_given_tensor_input_should_return_lfiltered_detections(
        self, mock_load_labels
    ):
        mock_cputfl = detectors.api_types[DetectorTypeEnum.cpu]

        TEST_DATA = np.zeros((1, 32, 32, 3), np.uint8)
        TEST_DETECT_RAW = [
            [2, 0.9, 5, 4, 3, 2],
            [1, 0.5, 8, 7, 6, 5],
            [0, 0.4, 2, 4, 8, 16],
        ]
        TEST_DETECT_RESULT = [
            ("label-3", 0.9, (5, 4, 3, 2)),
            ("label-2", 0.5, (8, 7, 6, 5)),
        ]
        TEST_LABEL_FILE = "/test_labels.txt"
        mock_load_labels.return_value = [
            "label-1",
            "label-2",
            "label-3",
            "label-4",
            "label-5",
        ]

        test_cfg = parse_obj_as(DetectorConfig, {"type": "cpu", "model": {}})
        test_cfg.model = ModelConfig()
        test_obj_detect = frigate.object_detection.base.LocalObjectDetector(
            detector_config=test_cfg,
            labels=TEST_LABEL_FILE,
        )

        mock_load_labels.assert_called_once_with(TEST_LABEL_FILE)

        mock_det_api = mock_cputfl.return_value
        mock_det_api.detect_raw.return_value = TEST_DETECT_RAW

        test_result = test_obj_detect.detect(tensor_input=TEST_DATA, threshold=0.5)

        mock_det_api.detect_raw.assert_called_once()
        assert (
            mock_det_api.detect_raw.call_args.kwargs["tensor_input"].shape
            == np.zeros((1, 32, 32, 3)).shape
        )
        assert test_result == TEST_DETECT_RESULT


class TestRawFrameHandThrough(unittest.TestCase):
    """A detector that normalises the raw frame itself is handed it untouched."""

    def _detector(self, api, input_tensor=InputTensorEnum.nchw, dtype=None):
        det = frigate.object_detection.base.BaseLocalDetector.__new__(
            frigate.object_detection.base.BaseLocalDetector
        )
        det.input_transform = tensor_transform(input_tensor)
        det.dtype = dtype or InputDTypeEnum.float
        det.detect_api = api
        return det

    def test_raw_frame_is_handed_through(self):
        frame = np.zeros((1, 8, 8, 3), np.uint8)
        det = self._detector(SimpleNamespace(accepts_uint8_nhwc=True))
        self.assertIs(det._transform_input(frame), frame)

    def test_other_detectors_still_get_the_float_tensor(self):
        frame = np.full((1, 8, 8, 3), 255, np.uint8)
        for api in (SimpleNamespace(accepts_uint8_nhwc=False), SimpleNamespace()):
            out = self._detector(api)._transform_input(frame)
            self.assertEqual(out.shape, (1, 3, 8, 8))
            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(float(out.max()), 1.0)

    def test_a_mock_detector_is_not_taken_for_support(self):
        # Mock answers every attribute with a truthy Mock
        frame = np.zeros((1, 8, 8, 3), np.uint8)
        self.assertEqual(
            self._detector(Mock())._transform_input(frame).dtype, np.float32
        )

    def test_only_the_normalised_nchw_input_is_replaced(self):
        frame = np.zeros((1, 8, 8, 3), np.uint8)
        api = SimpleNamespace(accepts_uint8_nhwc=True)
        nhwc = self._detector(api, input_tensor=InputTensorEnum.nhwc)._transform_input(
            frame
        )
        self.assertEqual((nhwc.shape, nhwc.dtype), ((1, 8, 8, 3), np.float32))
        denorm = self._detector(
            api, dtype=InputDTypeEnum.float_denorm
        )._transform_input(frame)
        self.assertEqual((denorm.shape, denorm.dtype), ((1, 3, 8, 8), np.float32))

    def test_detector_without_the_attribute_set_up_does_not_fail(self):
        det = self._detector(None)
        del det.detect_api
        frame = np.zeros((1, 8, 8, 3), np.uint8)
        self.assertEqual(det._transform_input(frame).dtype, np.float32)


class TestWriteDetections(unittest.TestCase):
    def setUp(self):
        self.out = np.full((20, 6), 7.0, dtype=np.float32)

    def test_empty_result_clears_buffer(self):
        frigate.object_detection.base.write_detections(
            self.out, np.zeros((0, 6), np.float32)
        )
        assert not self.out.any()

    def test_short_result_is_zero_padded(self):
        dets = np.ones((3, 6), np.float32)
        frigate.object_detection.base.write_detections(self.out, dets)
        assert (self.out[:3] == 1).all()
        assert not self.out[3:].any()

    def test_long_result_is_truncated(self):
        dets = np.ones((25, 6), np.float32)
        frigate.object_detection.base.write_detections(self.out, dets)
        assert (self.out == 1).all()
