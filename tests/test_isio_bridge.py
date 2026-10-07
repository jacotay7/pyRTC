"""pyrtc <-> ImageStreamIO bridge (#54); runs when ImageStreamIOWrap is installed."""

import time
import uuid

import numpy as np
import pytest

from pyrtc.isio_bridge import IsioBridge, _isio_module  # noqa: E402

try:
    ISIO = _isio_module()  # also preloads libImageStreamIO.so (#138)
except ImportError as exc:  # not installed, or unloadable
    pytest.skip(f"ImageStreamIO is not available: {exc}", allow_module_level=True)
from pyrtc.streams import open_stream  # noqa: E402
from testsupport import private_stream  # noqa: E402


def _name(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(1e-3)


@pytest.fixture
def isio_images():
    images = []
    yield images
    for image in images:
        try:
            image.destroy()
        except Exception:
            pass


def test_pyrtc_stream_mirrors_to_isio(isio_images):
    source = private_stream("isio_src", (3, 4), "float32")
    source.write(np.zeros((3, 4), dtype=np.float32))
    isio_name = _name("pyrtc_to")
    bridge = IsioBridge(
        {
            "name": "b",
            "direction": "to_isio",
            "isio_name": isio_name,
            "remove_on_close": True,
            "input_streams": {"input": source.name},
            "functions": ["mirror"],
        }
    )
    reader = ISIO.Image()
    assert reader.open(isio_name) == 0
    try:
        # pyrtc (height, width) = [y, x]; ISIO size = [x, y] (#162).
        assert [int(n) for n in reader.md.size] == [4, 3]
        bridge.start()
        frame = np.arange(12, dtype=np.float32).reshape(3, 4)
        source.write(frame)
        # The bridge's first read returns the current payload, so it may
        # mirror the zeros written above before this frame: wait for the
        # frame itself, not for any write. ISIO holds the same bytes with
        # its axes reversed, so pixel (x, y) is frame[y, x] on both sides.
        _wait_for(lambda: np.array_equal(np.asarray(reader.copy()), frame.T))
    finally:
        reader.close()
        bridge.close()
        source.close()
    assert not ISIO.Image().open(isio_name) == 0  # removed on close


def test_isio_stream_mirrors_into_pyrtc(isio_images):
    isio_name = _name("isio_from")
    writer = ISIO.Image()
    assert (
        writer.create(isio_name, np.asfortranarray(np.zeros((2, 5), np.uint16)), -1, 1, 10, 1) == 0
    )
    isio_images.append(writer)
    output = _name("from_isio")
    bridge = IsioBridge(
        {
            "name": "b",
            "direction": "from_isio",
            "isio_name": isio_name,
            "output_streams": {"output": output},
            "functions": ["mirror"],
        }
    )
    stream = open_stream(output, readonly=True)
    try:
        # ISIO size [2, 5] = [x, y] becomes a (5, 2) = [y, x] pyrtc stream.
        assert stream.shape == (5, 2) and stream.dtype == np.uint16
        bridge.start()
        frame = np.arange(10, dtype=np.uint16).reshape(2, 5)
        before = stream.count
        writer.write(np.asfortranarray(frame))
        _wait_for(lambda: stream.count > before)
        publication = stream.read_publication()
        np.testing.assert_array_equal(publication.payload, frame.T)
        assert publication.frame_id == int(writer.md.cnt0)  # ISIO cnt0 is the frame id
    finally:
        stream.close()
        bridge.close()


def test_existing_isio_stream_with_another_shape_is_refused(isio_images):
    isio_name = _name("isio_clash")
    other = ISIO.Image()
    other.create(isio_name, np.asfortranarray(np.zeros((5, 5), np.float32)), -1, 1, 10, 1)
    isio_images.append(other)
    source = private_stream("isio_src2", (3, 4), "float32")
    source.write(np.zeros((3, 4), dtype=np.float32))
    try:
        with pytest.raises(ValueError, match="exists with size"):
            IsioBridge(
                {
                    "name": "b",
                    "direction": "to_isio",
                    "isio_name": isio_name,
                    "input_streams": {"input": source.name},
                    "functions": ["mirror"],
                }
            )
    finally:
        source.close()


@pytest.mark.parametrize(
    ("conf", "match"),
    [
        ({"direction": "sideways", "isio_name": "x"}, "direction"),
        ({"direction": "to_isio"}, "isio_name"),
    ],
)
def test_config_is_validated(conf, match):
    with pytest.raises(ValueError, match=match):
        IsioBridge({"name": "b", "functions": [], **conf})


def test_missing_isio_stream_is_reported():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        IsioBridge(
            {
                "name": "b",
                "direction": "from_isio",
                "isio_name": _name("nowhere"),
                "output_streams": {"output": _name("o")},
                "functions": [],
            }
        )
