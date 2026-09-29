"""Facilitates communication between processes."""

import logging
import multiprocessing as mp
import threading
from collections.abc import Callable
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import zmq

from frigate.comms.base_communicator import Communicator

logger = logging.getLogger(__name__)

SOCKET_REP_REQ = "ipc:///tmp/cache/comms"


class InterProcessCommunicator(Communicator):
    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(SOCKET_REP_REQ)
        self.stop_event: MpEvent = mp.Event()

    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        """There is no communication back to the processes."""
        pass

    def subscribe(self, receiver: Callable) -> None:
        self._dispatcher = receiver
        self.reader_thread = threading.Thread(target=self.read)
        self.reader_thread.start()

    def read(self) -> None:
        while not self.stop_event.is_set():
            while True:  # load all messages that are queued
                has_message, _, _ = zmq.select([self.socket], [], [], 1)

                if not has_message:
                    break

                try:
                    raw = self.socket.recv_json(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(raw, list) and len(raw) == 2:
                    (topic, value) = raw
                    try:
                        response = self._dispatcher(topic, value)
                    except Exception:
                        # This is a background reader thread and it is the
                        # only thing servicing a REQ/REP socket. Any
                        # exception escaping the dispatcher used to kill
                        # the thread outright, after which every
                        # InterProcessRequestor.send_data() in every
                        # process blocked forever on recv_json() -- so a
                        # single unhandled topic wedged recording, review,
                        # stats, ffmpeg supervision and exports at once.
                        # A broad catch is correct here: staying alive
                        # matters more than any individual message.
                        logger.exception(
                            "Error dispatching inter-process message on topic %s",
                            topic,
                        )
                        response = None
                else:
                    # Wrong shape as well as wrong type: unpacking a list
                    # of any other length raised ValueError, which killed
                    # this thread the same way an escaping dispatcher
                    # exception did.
                    logger.warning(
                        "Discarding malformed ZMQ message (expected a 2-item list, got %s): %r",
                        type(raw).__name__,
                        raw,
                    )
                    response = None

                # REQ/REP is strictly alternating: the peer is already
                # blocked in recv, so a reply must go out on every path,
                # including the failure paths above.
                try:
                    self.socket.send_json(response if response is not None else [])
                except zmq.ZMQError:
                    break

    def stop(self) -> None:
        self.stop_event.set()
        self.reader_thread.join()
        self.socket.close(linger=0)
        self.context.destroy(linger=0)


# Safety net against a dead peer, not a latency budget. Most replies
# are near-instant, but some handlers do real work first --
# REQUEST_REGION_GRID runs get_camera_regions_grid, which queries the
# database and can be slow on a large install. Set this generously: the
# goal is to convert an unbounded hang into a bounded failure, and a
# false trip would degrade a working request.
REQUEST_TIMEOUT_MS = 30_000


class InterProcessRequestor:
    """Simplifies sending data to InterProcessCommunicator and getting a reply."""

    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self._make_socket()

    def _make_socket(self) -> zmq.Socket:
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, REQUEST_TIMEOUT_MS)
        socket.setsockopt(zmq.SNDTIMEO, REQUEST_TIMEOUT_MS)
        # Do not block termination on undelivered messages.
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(SOCKET_REP_REQ)
        return socket

    def send_data(self, topic: str, data: Any) -> Any:
        """Sends data and then waits for reply.

        Returns the peer's reply, or an empty string if the peer did not
        answer within REQUEST_TIMEOUT_MS.
        """
        try:
            self.socket.send_json((topic, data))
            return self.socket.recv_json()
        except zmq.Again:
            # Timed out. A REQ socket enforces strict send/recv
            # alternation, so after a missed reply it is stuck in the
            # wrong state and every later request on it would fail.
            # Discard it and reconnect -- the "lazy pirate" pattern.
            logger.warning(
                "Timed out waiting for inter-process reply on topic %s, resetting socket",
                topic,
            )
            self.socket.close(linger=0)
            self.socket = self._make_socket()
            return ""
        except zmq.ZMQError:
            return ""

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.destroy(linger=0)
