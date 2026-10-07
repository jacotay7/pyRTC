"""Mirror pyrtc streams to and from ImageStreamIO (milk / CACAO) shared memory.

ImageStreamIO (ISIO) is the shared-memory format of milk and CACAO, used at
Subaru/SCExAO, MagAO-X, COSMIC and others. An :class:`IsioBridge` copies one
stream in one direction, so pyrtc can use ISIO camera and DM drivers and milk
viewers, or feed a CACAO RTC:

.. code-block:: yaml

  isio_wfs:                       # pyrtc -> ISIO
    class_name: pyrtc.isio_bridge.IsioBridge
    direction: to_isio
    isio_name: pyrtc_wfs
    input_streams: {input: wfs}
    functions: [mirror]
  isio_dm:                        # ISIO -> pyrtc
    class_name: pyrtc.isio_bridge.IsioBridge
    direction: from_isio
    isio_name: dm00disp
    output_streams: {output: dm_from_cacao}
    functions: [mirror]

Or from the command line::

    python -m pyrtc.isio_bridge to-isio wfs pyrtc_wfs
    python -m pyrtc.isio_bridge from-isio dm00disp dm_from_cacao

Needs the ImageStreamIO Python module (``ImageStreamIOWrap``); build it with
``pip install git+https://github.com/milk-org/ImageStreamIO``. ISIO files live
in ``MILK_SHM_DIR`` (``/milk/shm`` or ``/tmp``).

Axes are reversed at the boundary, so images keep their orientation: pyrtc
arrays are row-major and indexed ``[y, x]`` (#162), while ISIO images are
column-major with ``size = [x, y]`` (``size[0]`` is the fastest axis, milk's
x). A pyrtc ``(height, width)`` stream becomes an ISIO image with
``size = [width, height]`` holding the same bytes, and an ISIO ``[nx, ny]``
image becomes a pyrtc ``(ny, nx)`` stream. (pyrtc 1.x kept the shape, which
matched milk while pyrtc streams were ``(width, height)``.) The ISIO-to-pyrtc direction publishes each new ISIO frame (its
``cnt0`` becomes the pyrtc ``frame_id``) within about ``poll_interval``.
Waiting uses non-blocking semaphore polls, because the ISIO module's blocking
waits hold the GIL and would stall every other pyrtc thread.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pyrtc.component import Component
from pyrtc.logging_utils import get_logger
from pyrtc.streams import create_stream, open_stream
from pyrtc.utils import require_optional

logger = get_logger(__name__)

DIRECTIONS = ("to_isio", "from_isio")


def _preload_isio_library() -> None:
    """Load ``libImageStreamIO.so`` from ``sys.path`` so the wrapper can link to it.

    ``pip install`` puts the library next to ``ImageStreamIOWrap`` in
    site-packages, but the extension's RPATH does not include ``$ORIGIN``, so
    importing it fails with "libImageStreamIO.so: cannot open shared object
    file" unless the library is already loaded (#138).
    """

    import ctypes
    import os
    import sys

    for entry in sys.path:
        candidate = os.path.join(entry or ".", "libImageStreamIO.so")
        if os.path.isfile(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            return


def _isio_module():
    _preload_isio_library()
    try:
        return require_optional("ImageStreamIOWrap", "isio", "The ImageStreamIO bridge")
    except ImportError as exc:
        raise ImportError(
            "The ImageStreamIO bridge needs the ImageStreamIOWrap module; install it with "
            "pip install git+https://github.com/milk-org/ImageStreamIO"
        ) from exc


def _isio_dtype(image) -> np.dtype:
    """Numpy dtype of an open ISIO image (via a copy of its data)."""

    return np.asarray(image.copy()).dtype


class IsioBridge(Component):
    """Copy one stream between pyrtc (pyshmem) and ImageStreamIO, in one direction."""

    EXTRA_CONFIG_KEYS = (
        "direction",
        "isio_name",
        "poll_interval",
        "wait_slice",
        "num_semaphores",
        "remove_on_close",
    )

    def __init__(self, conf) -> None:
        self.direction = str(conf.get("direction", "")).lower().replace("-", "_")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"isio bridge: direction must be one of {DIRECTIONS}")
        self.isio_name = str(conf.get("isio_name") or "")
        if not self.isio_name:
            raise ValueError("isio bridge: set 'isio_name'")
        self.poll_interval = float(conf.get("poll_interval", 1e-4))
        self.wait_slice = float(conf.get("wait_slice", 0.1))
        self.num_semaphores = int(conf.get("num_semaphores", 10))
        # ISIO streams usually outlive their producer so consumers can keep
        # attaching; remove one this bridge created only when asked.
        self.remove_on_close = bool(conf.get("remove_on_close", False))
        self._created = False
        self._isio = _isio_module()
        super().__init__(conf)
        self._image = self._isio.Image()
        if self.direction == "to_isio":
            self._setup_to_isio()
        else:
            self._setup_from_isio()
        self.logger.info(
            "ISIO bridge %s: %s %s %s",
            self.direction,
            self.stream_name,
            "->" if self.direction == "to_isio" else "<-",
            self.isio_name,
        )

    # -- pyrtc -> ISIO --------------------------------------------------------

    def _setup_to_isio(self) -> None:
        self.stream_name = self.input_stream_name("input")
        self.register_input_stream("input", open_stream(self.stream_name, readonly=True))
        frame = np.asarray(self.read_stream("input", block=False))
        if hasattr(frame, "detach"):  # GPU stream
            frame = frame.detach().cpu().numpy()
        self.shape, self.dtype = tuple(frame.shape), np.dtype(frame.dtype)
        isio_shape = self.shape[::-1]
        if self._image.open(self.isio_name) == 0:
            existing_shape = tuple(int(n) for n in self._image.md.size)
            existing_dtype = _isio_dtype(self._image)
            if existing_shape != isio_shape or existing_dtype != self.dtype:
                raise ValueError(
                    f"ISIO stream {self.isio_name!r} exists with size {existing_shape} "
                    f"{existing_dtype}; the pyrtc stream {self.shape} {self.dtype} needs "
                    f"size {isio_shape}. Remove it or pick another isio_name."
                )
        else:
            self._image = self._isio.Image()
            error = self._image.create(
                self.isio_name, _to_isio_layout(frame), -1, 1, self.num_semaphores, 1
            )
            if error:
                raise RuntimeError(f"ISIO create({self.isio_name!r}) failed with {error}")
            self._created = True

    def _write_isio(self, frame) -> None:
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        self._image.write(_to_isio_layout(frame, self.dtype))

    # -- ISIO -> pyrtc --------------------------------------------------------

    def _setup_from_isio(self) -> None:
        if self._image.open(self.isio_name) != 0:
            raise FileNotFoundError(f"ISIO stream {self.isio_name!r} does not exist")
        frame = _from_isio_layout(self._image.copy())
        self.shape, self.dtype = tuple(frame.shape), frame.dtype
        self.stream_name = self.output_stream_name("output")
        output = create_stream(self.stream_name, self.shape, self.dtype)
        self.register_output_stream("output", output)
        self._semaphore = int(self._image.getsemwaitindex(0))
        if self._semaphore < 0:
            raise RuntimeError(f"ISIO stream {self.isio_name!r} has no free semaphore")
        self._image.semflush(self._semaphore)
        self._publish(frame)

    def _publish(self, frame) -> None:
        self.frame_id = int(self._image.md.cnt0)
        self.write_stream("output", frame)

    def _wait_isio(self) -> bool:
        """Poll for a new ISIO frame for up to ``wait_slice`` seconds."""

        deadline = time.monotonic() + self.wait_slice
        while self._image.semtrywait(self._semaphore) != 0:
            if time.monotonic() > deadline or not self.running:
                return False
            time.sleep(self.poll_interval)
        self._image.semflush(self._semaphore)  # coalesce writes we fell behind on
        return True

    # -- worker ----------------------------------------------------------------

    def mirror(self):
        """Worker function: forward the next frame in the configured direction."""

        if self.direction == "to_isio":
            try:
                frame = self.read_stream("input", timeout=self.wait_slice)
            except TimeoutError:
                return
            self._write_isio(frame)
        elif self._wait_isio():
            self._publish(_from_isio_layout(self._image.copy()))

    def close(self, *args, **kwargs):
        try:
            super().close(*args, **kwargs)
        finally:
            image = getattr(self, "_image", None)
            if image is not None:
                try:
                    if self._created and self.remove_on_close:
                        image.destroy()  # the creator's destroy also removes the file
                    else:
                        image.close()
                except Exception:
                    pass
                self._image = None


def _to_isio_layout(frame, dtype=None) -> np.ndarray:
    """A pyrtc ``[y, x]`` array as the column-major ``[x, y]`` array ISIO stores.

    The transpose of a row-major array is already column-major, so this is
    the same memory (ISIO's Python write only accepts that layout).
    """

    return np.asfortranarray(np.asarray(frame).T, dtype=dtype)


def _from_isio_layout(image) -> np.ndarray:
    """An ISIO ``[x, y]`` array as a row-major pyrtc ``[y, x]`` array."""

    return np.ascontiguousarray(np.asarray(image).T)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mirror one stream between pyrtc and ImageStreamIO."
    )
    parser.add_argument("direction", choices=("to-isio", "from-isio"))
    parser.add_argument("source", help="pyrtc stream (to-isio) or ISIO stream (from-isio)")
    parser.add_argument("target", help="ISIO stream (to-isio) or pyrtc stream (from-isio)")
    parser.add_argument("--poll-interval", type=float, default=1e-4)
    args = parser.parse_args(argv)
    conf = {"name": "isio_bridge", "functions": ["mirror"], "poll_interval": args.poll_interval}
    if args.direction == "to-isio":
        conf.update(
            direction="to_isio", isio_name=args.target, input_streams={"input": args.source}
        )
    else:
        conf.update(
            direction="from_isio", isio_name=args.source, output_streams={"output": args.target}
        )
    bridge = IsioBridge(conf)
    bridge.start()
    print(f"Mirroring {args.source} -> {args.target}; Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
