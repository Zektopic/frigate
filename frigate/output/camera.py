"""Handle outputting individual cameras via jsmpeg."""

import logging
import queue
import subprocess as sp
import threading
import time
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from frigate.config import CameraConfig, FfmpegConfig, FrigateConfig
from frigate.output.ws_auth import ws_has_camera_access

logger = logging.getLogger(__name__)


class FFMpegConverter(threading.Thread):
    def __init__(
        self,
        camera: str,
        ffmpeg: FfmpegConfig,
        input_queue: queue.Queue,
        stop_event: MpEvent,
        in_width: int,
        in_height: int,
        out_width: int,
        out_height: int,
        quality: int,
    ):
        super().__init__(name=f"{camera}_output_converter")
        self.camera = camera
        self.input_queue = input_queue
        self.stop_event = stop_event

        ffmpeg_cmd = [
            ffmpeg.ffmpeg_path,
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-video_size",
            f"{in_width}x{in_height}",
            "-i",
            "pipe:",
            "-threads",
            "1",
            "-f",
            "mpegts",
            "-s",
            f"{out_width}x{out_height}",
            "-codec:v",
            "mpeg1video",
            "-q",
            f"{quality}",
            "-bf",
            "0",
            "pipe:",
        ]

        self.process = sp.Popen(
            ffmpeg_cmd,
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
            stdin=sp.PIPE,
            start_new_session=True,
        )

    def __write(self, b: bytes) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(b)

    def read(self, length: int) -> Any:
        try:
            return self.process.stdout.read1(length)  # type: ignore[union-attr]
        except ValueError:
            return False

    def exit(self) -> None:
        self.process.terminate()

        try:
            self.process.communicate(timeout=30)
        except sp.TimeoutExpired:
            self.process.kill()
            self.process.communicate()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(True, timeout=1)
                self.__write(frame)
            except queue.Empty:
                pass

        self.exit()


class BroadcastThread(threading.Thread):
    def __init__(
        self,
        camera: str,
        converter: FFMpegConverter,
        websocket_server: Any,
        stop_event: MpEvent,
        config: FrigateConfig,
    ):
        super().__init__()
        self.camera = camera
        self.converter = converter
        self.websocket_server = websocket_server
        self.stop_event = stop_event
        self.config = config

    def run(self) -> None:
        while not self.stop_event.is_set():
            buf = self.converter.read(65536)
            if buf:
                manager = self.websocket_server.manager
                with manager.lock:
                    websockets = manager.websockets.copy()
                    ws_iter = iter(websockets.values())

                for ws in ws_iter:
                    if (
                        not ws.terminated
                        and ws.environ["PATH_INFO"] == f"/{self.camera}"
                        and ws_has_camera_access(ws, self.camera, self.config)
                    ):
                        try:
                            ws.send(buf, binary=True)
                        except ValueError:
                            pass
                        except (BrokenPipeError, ConnectionResetError) as e:
                            logger.debug(f"Websocket unexpectedly closed {e}")
            elif self.converter.process.poll() is not None:
                break


class _AnyStopEvent:
    """Report stopped when either the process-wide or session event is set."""

    def __init__(self, *events: Any) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class JsmpegCamera:
    """jsmpeg live stream for one camera.

    The mpeg1 encoder (an ffmpeg process plus two threads) only runs while
    someone is watching: it is started by the first frame written for a
    connected client and stopped after ``idle_timeout`` seconds without
    frames. An idle encoder per camera otherwise costs ~20 MB RSS each.
    """

    idle_timeout = 60.0

    def __init__(
        self,
        config: CameraConfig,
        frigate_config: FrigateConfig,
        stop_event: MpEvent,
        websocket_server: Any,
    ) -> None:
        self.config = config
        self.frigate_config = frigate_config
        self.stop_event = stop_event
        self.websocket_server = websocket_server
        self.input: queue.Queue[bytes] = queue.Queue(maxsize=config.detect.fps)
        self.converter: FFMpegConverter | None = None
        self.broadcaster: BroadcastThread | None = None
        self._session_stop: threading.Event | None = None
        self._last_write = 0.0

    @property
    def running(self) -> bool:
        return (
            self.converter is not None
            and self._session_stop is not None
            and not self._session_stop.is_set()
            and self.converter.process.poll() is None
        )

    def _start(self) -> None:
        # make sure a crashed previous session's threads wind down
        self.shutdown()

        # drop frames queued for a previous session
        while not self.input.empty():
            try:
                self.input.get_nowait()
            except queue.Empty:
                break

        width = int(
            self.config.live.height
            * (self.config.frame_shape[1] / self.config.frame_shape[0])
        )
        self._session_stop = threading.Event()
        session_stop = _AnyStopEvent(self.stop_event, self._session_stop)
        self.converter = FFMpegConverter(
            self.config.name or "",
            self.config.ffmpeg,
            self.input,
            session_stop,  # type: ignore[arg-type]
            self.config.frame_shape[1],
            self.config.frame_shape[0],
            width,
            self.config.live.height,
            self.config.live.quality,
        )
        self.broadcaster = BroadcastThread(
            self.config.name or "",
            self.converter,
            self.websocket_server,
            session_stop,  # type: ignore[arg-type]
            self.frigate_config,
        )
        self.converter.start()
        self.broadcaster.start()
        logger.debug("Started jsmpeg encoder for %s", self.config.name)

    def write_frame(self, frame_bytes: bytes, now: float | None = None) -> None:
        self._last_write = time.monotonic() if now is None else now

        if not self.running:
            self._start()

        try:
            self.input.put_nowait(frame_bytes)
        except queue.Full:
            # drop frames if queue is full
            pass

    def check_idle(self, now: float | None = None) -> None:
        """Stop the encoder once no client has been fed frames for a while."""
        if self._session_stop is None or self._session_stop.is_set():
            return

        now = time.monotonic() if now is None else now
        if now - self._last_write >= self.idle_timeout:
            logger.debug("Stopping idle jsmpeg encoder for %s", self.config.name)
            # threads exit on their own within ~1s; don't block the caller
            self._session_stop.set()

    def shutdown(self) -> None:
        """Signal the current encoder session (if any) to stop."""
        if self._session_stop is not None:
            self._session_stop.set()

    def stop(self) -> None:
        if self.converter is not None:
            self.converter.join()
        if self.broadcaster is not None:
            self.broadcaster.join()
