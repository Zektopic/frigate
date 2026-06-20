import unittest
from unittest.mock import MagicMock, patch

from frigate.config import FrigateConfig
from frigate.config.camera.ffmpeg import FFMPEG_INPUT_ARGS_DEFAULT
from frigate.ffmpeg_presets import LibvaGpuSelector, parse_preset_input


class TestFfmpegPresets(unittest.TestCase):
    def setUp(self):
        self.default_ffmpeg = {
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "back": {
                    "ffmpeg": {
                        "inputs": [
                            {
                                "path": "rtsp://10.0.0.1:554/video",
                                "roles": ["detect"],
                            }
                        ],
                        "output_args": {
                            "detect": "-f rawvideo -pix_fmt yuv420p",
                            "record": "-f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1 -strftime 1 -c copy -an",
                        },
                    },
                    "detect": {
                        "height": 1080,
                        "width": 1920,
                        "fps": 5,
                    },
                    "record": {
                        "enabled": True,
                    },
                    "name": "back",
                }
            },
        }

    def test_default_ffmpeg(self):
        FrigateConfig(**self.default_ffmpeg)

    def test_ffmpeg_hwaccel_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "preset-rpi-64-h264"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-rpi-64-h264" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert "-c:v:1 h264_v4l2m2m" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "-other-hwaccel args"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-other-hwaccel args" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_scale_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "preset-nvidia-h264"
        )
        self.default_ffmpeg["cameras"]["back"]["detect"] = {
            "height": 1920,
            "width": 2560,
            "fps": 10,
        }
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-nvidia-h264" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert "fps=10,scale_cuda=w=2560:h=1920,hwdownload,format=nv12" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_default_ffmpeg_input_arg_preset(self):
        frigate_config = FrigateConfig(**self.default_ffmpeg)

        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = (
            "preset-rtsp-generic"
        )
        frigate_preset_config = FrigateConfig(**self.default_ffmpeg)
        assert (
            # Ignore global and user_agent args in comparison
            frigate_preset_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
            == frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
        )

    def test_ffmpeg_input_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = (
            "preset-rtmp-generic"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-rtmp-generic" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert (" ".join(parse_preset_input("preset-rtmp-generic", 5))) in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_input_args_as_string(self):
        # Strip user_agent args here to avoid handling quoting issues
        defaultArgsList = parse_preset_input(FFMPEG_INPUT_ARGS_DEFAULT, 5)[2::]
        argsString = " ".join(defaultArgsList) + ' -some "arg with space"'
        argsList = defaultArgsList + ["-some", "arg with space"]
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = argsString
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert set(argsList).issubset(
            frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
        )

    def test_ffmpeg_input_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = "-some inputs"
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-some inputs" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_output_record_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["output_args"]["record"] = (
            "preset-record-generic-audio-aac"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-record-generic-audio-aac" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert "-c:v copy -c:a aac" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_output_record_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["output_args"]["record"] = (
            "-some output -segment_time 10"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-some output" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )


class TestLibvaGpuSelector(unittest.TestCase):
    def setUp(self):
        self.selector = LibvaGpuSelector()
        # Reset _valid_gpus before each test
        LibvaGpuSelector._valid_gpus = None

    def test_nvidia_preset(self):
        self.assertEqual(self.selector.get_gpu_arg("preset-nvidia-h264", 2), "2")
        self.assertIsNone(self.selector._valid_gpus)  # Should not be initialized

    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_no_dri_dir(self, mock_exists):
        mock_exists.return_value = False
        self.assertEqual(self.selector.get_gpu_arg("preset-vaapi", 0), "")
        self.assertEqual(self.selector._valid_gpus, [])

    @patch("frigate.ffmpeg_presets.os.listdir")
    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_no_render_devices(self, mock_exists, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ["card0"]
        self.assertEqual(
            self.selector.get_gpu_arg("preset-vaapi", 0), "/dev/dri/renderD128"
        )
        self.assertEqual(self.selector._valid_gpus, ["/dev/dri/renderD128"])

    @patch("frigate.ffmpeg_presets.os.listdir")
    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_one_render_device(self, mock_exists, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ["card0", "renderD128"]
        self.assertEqual(
            self.selector.get_gpu_arg("preset-vaapi", 0), "/dev/dri/renderD128"
        )
        self.assertEqual(self.selector._valid_gpus, ["/dev/dri/renderD128"])

    @patch("frigate.ffmpeg_presets.vainfo_hwaccel")
    @patch("frigate.ffmpeg_presets.os.listdir")
    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_multiple_render_devices_all_valid(
        self, mock_exists, mock_listdir, mock_vainfo
    ):
        mock_exists.return_value = True
        mock_listdir.return_value = ["card0", "renderD128", "renderD129"]
        mock_vainfo.return_value.returncode = 0

        self.assertEqual(
            self.selector.get_gpu_arg("preset-vaapi", 1), "/dev/dri/renderD129"
        )
        self.assertEqual(
            self.selector._valid_gpus, ["/dev/dri/renderD128", "/dev/dri/renderD129"]
        )

    @patch("frigate.ffmpeg_presets.vainfo_hwaccel")
    @patch("frigate.ffmpeg_presets.os.listdir")
    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_multiple_render_devices_some_valid(
        self, mock_exists, mock_listdir, mock_vainfo
    ):
        mock_exists.return_value = True
        mock_listdir.return_value = ["card0", "renderD128", "renderD129"]

        def vainfo_side_effect(device_name):
            mock_result = MagicMock()
            if device_name == "renderD128":
                mock_result.returncode = 1  # Invalid
            else:
                mock_result.returncode = 0  # Valid
            return mock_result

        mock_vainfo.side_effect = vainfo_side_effect

        self.assertEqual(
            self.selector.get_gpu_arg("preset-vaapi", 0), "/dev/dri/renderD129"
        )
        self.assertEqual(self.selector._valid_gpus, ["/dev/dri/renderD129"])

    @patch("frigate.ffmpeg_presets.vainfo_hwaccel")
    @patch("frigate.ffmpeg_presets.os.listdir")
    @patch("frigate.ffmpeg_presets.os.path.exists")
    def test_invalid_gpu_index(self, mock_exists, mock_listdir, mock_vainfo):
        mock_exists.return_value = True
        mock_listdir.return_value = ["card0", "renderD128", "renderD129"]
        mock_vainfo.return_value.returncode = 0

        # Index 5 is out of bounds (only 2 devices)
        self.assertEqual(
            self.selector.get_gpu_arg("preset-vaapi", 5), "/dev/dri/renderD128"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
