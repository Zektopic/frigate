import unittest

from frigate.config.camera.ffmpeg import CameraFfmpegConfig, CameraInput
from frigate.const import AUDIO_FORMAT, AUDIO_SAMPLE_RATE
from frigate.events.audio import get_ffmpeg_command


class TestAudioFfmpegCommand(unittest.TestCase):
    def test_get_ffmpeg_command_default_args(self):
        ffmpeg_config = CameraFfmpegConfig(
            inputs=[CameraInput(path="rtsp://test", roles=["audio", "detect"])]
        )
        command = get_ffmpeg_command(ffmpeg_config)

        # Verify global args and default input args are used
        self.assertIn("-hide_banner", command)
        self.assertIn("-loglevel", command)
        self.assertIn("warning", command)
        # Default FFMPEG_INPUT_ARGS_DEFAULT is "preset-rtsp-generic" which parses to specific args
        # But `parse_preset_input(ffmpeg.input_args, 1)` -> `-avoid_negative_ts make_zero -fflags +genpts+discardcorrupt -rtsp_transport tcp -stimeout 5000000 -use_wallclock_as_timestamps 1`
        self.assertIn("-avoid_negative_ts", command)
        self.assertIn("make_zero", command)

        # Verify output args for audio
        self.assertIn("-vn", command)
        self.assertIn("-f", command)
        self.assertIn(str(AUDIO_FORMAT), command)
        self.assertIn("-ar", command)
        self.assertIn(str(AUDIO_SAMPLE_RATE), command)
        self.assertIn("-ac", command)
        self.assertIn("1", command)
        self.assertIn("pipe:", command)
        self.assertIn("rtsp://test", command)

    def test_get_ffmpeg_command_custom_input_args(self):
        # Override input args on the camera input
        ffmpeg_config = CameraFfmpegConfig(
            inputs=[
                CameraInput(
                    path="rtsp://test",
                    roles=["audio", "detect"],
                    input_args=["-custom", "arg1"],
                )
            ]
        )
        command = get_ffmpeg_command(ffmpeg_config)

        # Custom input args on the input should take precedence over the global config's input args
        self.assertIn("-custom", command)
        self.assertIn("arg1", command)

        # Should not contain the preset default
        self.assertNotIn("-avoid_negative_ts", command)

    def test_get_ffmpeg_command_global_args(self):
        # Override global args
        ffmpeg_config = CameraFfmpegConfig(
            global_args="-custom_global arg",
            inputs=[CameraInput(path="rtsp://test", roles=["audio", "detect"])],
        )
        command = get_ffmpeg_command(ffmpeg_config)

        self.assertIn("-custom_global", command)
        self.assertIn("arg", command)
        # The default global args shouldn't be there anymore
        self.assertNotIn("-hide_banner", command)

    def test_get_ffmpeg_command_with_preset(self):
        # Override input args with a preset string
        ffmpeg_config = CameraFfmpegConfig(
            inputs=[
                CameraInput(
                    path="rtsp://test",
                    roles=["audio", "detect"],
                    input_args="preset-rtsp-udp",
                )
            ]
        )
        command = get_ffmpeg_command(ffmpeg_config)

        # preset-rtsp-udp contains specific values like -rtsp_transport udp
        self.assertIn("udp", command)
        self.assertNotIn("tcp", command)
