"""GPU stream tests for pyrtc's pyshmem-backed transport.

These tests require a CUDA-capable torch installation, which GitHub-hosted
CI runners do not provide: every test is skipped automatically when CUDA is
unavailable, and CI can additionally deselect the lane with ``-m "not gpu"``.
The lane is intended to run on the GPU lab machine before releases.
"""

import uuid

import numpy as np
import pytest

import pyshmem
from pyrtc.streams import clear_shms, create_stream, open_stream

CUDA_AVAILABLE = pyshmem.gpu_available()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available"),
]


@pytest.fixture
def stream_name():
    name = f"pyrtc_gpu_{uuid.uuid4().hex[:10]}"
    yield name
    clear_shms([name])


def test_create_stream_gpu_producer_has_cpu_mirror(stream_name):
    shm = create_stream(stream_name, (8,), np.float32, gpu_device="cuda:0")
    try:
        assert shm.gpu_enabled
        assert shm.cpu_mirror, "pyrtc GPU streams must always carry a CPU mirror"
        shm.write(np.arange(8, dtype=np.float32))
    finally:
        shm.close()


def test_cpu_consumer_reads_numpy_from_gpu_stream(stream_name):
    producer = create_stream(stream_name, (8,), np.float32, gpu_device="cuda:0")
    try:
        producer.write(np.arange(8, dtype=np.float32))

        consumer = open_stream(stream_name)
        try:
            payload = consumer.read()
            assert isinstance(payload, np.ndarray)
            assert np.allclose(payload, np.arange(8, dtype=np.float32))
        finally:
            consumer.close()
    finally:
        producer.close()


def test_gpu_consumer_reads_cuda_tensor(stream_name):
    import torch

    producer = create_stream(stream_name, (8,), np.float32, gpu_device="cuda:0")
    try:
        producer.write(np.arange(8, dtype=np.float32))

        consumer = open_stream(stream_name, gpu_device="cuda:0")
        try:
            payload = consumer.read()
            assert isinstance(payload, torch.Tensor)
            assert payload.is_cuda
            assert torch.allclose(payload.cpu(), torch.arange(8, dtype=torch.float32))
        finally:
            consumer.close()
    finally:
        producer.close()


def test_gpu_write_accepts_cuda_tensor(stream_name):
    import torch

    producer = create_stream(stream_name, (4,), np.float32, gpu_device="cuda:0")
    try:
        producer.write(torch.full((4,), 3.0, device="cuda:0"))

        mirror = open_stream(stream_name)
        try:
            assert np.allclose(mirror.read(), 3.0)
        finally:
            mirror.close()
    finally:
        producer.close()


def test_unsupported_gpu_dtype_falls_back_to_cpu(stream_name):
    # uint16 (the wfs_raw dtype) has no torch equivalent in pyshmem's GPU map.
    shm = create_stream(stream_name, (4,), np.uint16, gpu_device="cuda:0")
    try:
        assert not shm.gpu_enabled
        shm.write(np.arange(4, dtype=np.uint16))
        assert np.array_equal(shm.read(), np.arange(4, dtype=np.uint16))
    finally:
        shm.close()


def test_gpu_stream_read_new_with_out_buffer_on_mirror(stream_name):
    import threading
    import time

    producer = create_stream(stream_name, (4,), np.float32, gpu_device="cuda:0")
    try:
        producer.write(np.zeros(4, dtype=np.float32))
        consumer = open_stream(stream_name)
        buffer = np.empty(4, dtype=np.float32)
        try:
            writer = threading.Thread(
                target=lambda: (
                    time.sleep(0.05),
                    producer.write(np.arange(4, dtype=np.float32)),
                )
            )
            writer.start()
            result = consumer.read_new(timeout=5.0, out=buffer)
            writer.join()

            assert result is buffer
            assert np.allclose(buffer, np.arange(4, dtype=np.float32))
        finally:
            consumer.close()
    finally:
        producer.close()
