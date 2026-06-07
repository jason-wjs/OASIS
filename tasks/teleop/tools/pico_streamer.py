"""
Stream a single Isaac Sim camera (e.g. front_camera) to the PICO XRoboToolkit
Unity client over the same wire format used by XRoboToolkit-Orin-Video-Sender.

Wire format (verified from main_web_gst.cpp + network_helper.hpp in
github.com/XR-Robotics/XRoboToolkit-Orin-Video-Sender):

    role:   TCP client; we connect to <PICO_IP>:<port> (default 12345).
            The PICO Unity client opens a TCP server on that port.
    frame:  per encoded H.264 access unit ->
                [4-byte big-endian uint32 length][Annex-B NAL bytes]
    codec:  H.264, Annex-B, alignment=au, stream-format=byte-stream.
            SPS/PPS repeated on every IDR so a mid-stream connect resyncs.

Default output resolution 2560x720@60 mono-as-stereo (front frame duplicated
left/right) so the existing `ZEDMINI` profile in the APK's video_source.yml
(2560x720, 4 Mbps, side-by-side) can be reused without modification.

Usage from teleop code:

    from tasks.teleop.tools.pico_streamer import PicoStreamer
    streamer = PicoStreamer(host="192.168.1.176")
    streamer.start()
    ...
    streamer.push(front_rgb_uint8)   # H x W x 3 (or 4) numpy array
    ...
    streamer.stop()

Standalone smoke test (no Isaac Sim, generates a moving rectangle):

    python -m tasks.teleop.tools.pico_streamer --host 192.168.1.176
"""

from __future__ import annotations

import argparse
import logging
import queue
import socket
import struct
import threading
import time
from fractions import Fraction
from typing import Optional

import av
import cv2
import numpy as np

logger = logging.getLogger("pico_streamer")


class PicoStreamer:
    def __init__(
        self,
        host: str,
        port: int = 12345,
        out_w: int = 2560,
        out_h: int = 720,
        fps: int = 50,
        bitrate: int = 4_000_000,
        gop: int = 30,
        reconnect_interval: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.out_w = out_w
        self.out_h = out_h
        self.fps = fps
        self.bitrate = bitrate
        self.gop = gop
        self.reconnect_interval = reconnect_interval

        self._eye_w = out_w // 2
        self._eye_h = out_h
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --------- public API ---------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve_forever, name="pico_streamer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)  # unblock _next_frame
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def push(self, rgb: np.ndarray) -> None:
        """Non-blocking; drops the oldest frame if the queue is full."""
        if self._stop.is_set():
            return
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(rgb)
        except queue.Full:
            pass

    # --------- internals ---------

    def _next_frame(self, timeout: float) -> Optional[np.ndarray]:
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return item  # may be None if stop() was called

    def _stitch_sbs(self, rgb: np.ndarray) -> np.ndarray:
        """Resize to per-eye, letterbox to 16:9, duplicate to side-by-side."""
        if rgb.ndim != 3:
            raise ValueError(f"expected HxWxC, got shape {rgb.shape}")
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        # preserve aspect: fit inside (eye_w, eye_h), pad black around
        scale = min(self._eye_w / w, self._eye_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        eye = np.zeros((self._eye_h, self._eye_w, 3), dtype=np.uint8)
        x0 = (self._eye_w - new_w) // 2
        y0 = (self._eye_h - new_h) // 2
        eye[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return np.concatenate([eye, eye], axis=1)  # left == right

    def _open_codec(self) -> av.CodecContext:
        codec = av.CodecContext.create("libx264", "w")
        codec.width = self.out_w
        codec.height = self.out_h
        codec.pix_fmt = "yuv420p"
        codec.framerate = Fraction(self.fps, 1)
        codec.time_base = Fraction(1, self.fps)
        codec.bit_rate = self.bitrate
        codec.gop_size = self.gop
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "x264-params": f"repeat-headers=1:annexb=1:keyint={self.gop}:min-keyint={self.gop}:scenecut=0",
        }
        codec.open()
        return codec

    def _serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                logger.info("connecting to PICO at %s:%d", self.host, self.port)
                sock = socket.create_connection(
                    (self.host, self.port), timeout=5.0
                )
                sock.settimeout(None)
                # disable Nagle for low latency
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as e:
                logger.warning("connect failed (%s); retry in %.1fs", e, self.reconnect_interval)
                if self._stop.wait(self.reconnect_interval):
                    return
                continue

            logger.info("connected, starting encoder")
            codec = self._open_codec()
            try:
                self._stream_loop(sock, codec)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logger.warning("stream error: %s", e)
            finally:
                # flush + close encoder
                try:
                    for _ in codec.encode(None):
                        pass
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass

            if self._stop.is_set():
                return
            # brief backoff before reconnecting
            self._stop.wait(self.reconnect_interval)

    def _stream_loop(self, sock: socket.socket, codec: av.CodecContext) -> None:
        # pts is in 1/fps ticks (codec.time_base = 1/fps), derived from wall clock
        # so push() cadence != fps doesn't drift the timestamps.
        t0 = time.monotonic()
        last_pts = -1
        while not self._stop.is_set():
            rgb = self._next_frame(timeout=1.0)
            if rgb is None:
                if self._stop.is_set():
                    return
                continue  # idle: keep connection, wait for next frame

            sbs = self._stitch_sbs(rgb)
            frame = av.VideoFrame.from_ndarray(sbs, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            pts = int((time.monotonic() - t0) * self.fps)
            if pts <= last_pts:
                pts = last_pts + 1
            last_pts = pts
            frame.pts = pts

            for packet in codec.encode(frame):
                payload = bytes(packet)
                if not payload:
                    continue
                header = struct.pack(">I", len(payload))
                sock.sendall(header + payload)


# ---------- standalone smoke test ----------


def _smoke_test(host: str, port: int, fps: int, duration: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    streamer = PicoStreamer(host=host, port=port, fps=fps)
    streamer.start()

    W, H = 224, 224
    n_frames = int(fps * duration)
    period = 1.0 / fps
    t0 = time.time()
    try:
        for i in range(n_frames):
            img = np.full((H, W, 3), 30, dtype=np.uint8)
            # moving white rectangle
            x = (i * 4) % (W - 40)
            y = (i * 3) % (H - 40)
            img[y : y + 40, x : x + 40] = (255, 255, 255)
            # red border so we can tell orientation
            img[0:4, :] = (255, 0, 0)
            img[-4:, :] = (255, 0, 0)
            img[:, 0:4] = (255, 0, 0)
            img[:, -4:] = (255, 0, 0)
            streamer.push(img)
            target = t0 + (i + 1) * period
            sleep_for = target - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        streamer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PICO streamer smoke test")
    parser.add_argument("--host", required=True, help="PICO headset IP")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    _smoke_test(args.host, args.port, args.fps, args.duration)
