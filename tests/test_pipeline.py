import numpy as np

from pyRTC import Pipeline as pipeline


def _cleanup_shm(obj):
    try:
        obj.close()
    except Exception:
        pass
    try:
        obj.shm.unlink()
    except Exception:
        pass
    if hasattr(obj, "metadataShm"):
        try:
            obj.metadataShm.close()
        except Exception:
            pass
        try:
            obj.metadataShm.unlink()
        except Exception:
            pass


def test_normalize_gpu_device_falls_back(monkeypatch):
    monkeypatch.setattr(pipeline, "TORCH_AVAILABLE", False)
    assert pipeline.normalize_gpu_device("cuda:0", "ctx") is None


def test_image_shm_cpu_read_write(unique_name):
    name = unique_name("img")
    shm = pipeline.ImageSHM(name, (4, 3), np.float32, gpuDevice=None, consumer=False)
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert shm.write(arr) == 1
    out = shm.read_noblock()
    assert np.array_equal(out, arr)
    _cleanup_shm(shm)


def test_init_existing_shm(unique_name):
    name = unique_name("existing")
    prod = pipeline.ImageSHM(name, (2, 2), np.int32, consumer=False)
    prod.write(np.array([[1, 2], [3, 4]], dtype=np.int32))
    cons, dims, dtype = pipeline.initExistingShm(name)
    out = cons.read_noblock()
    assert dims == [2, 2]
    assert dtype == np.dtype(np.int32)
    assert np.array_equal(out, np.array([[1, 2], [3, 4]], dtype=np.int32))
    _cleanup_shm(cons)
    _cleanup_shm(prod)


def test_clear_shms(unique_name):
    name = unique_name("clear")
    shm = pipeline.ImageSHM(name, (1,), np.uint8, consumer=False)
    _ = pipeline.ImageSHM(name + "_meta", (pipeline.ImageSHM.METADATA_SIZE,), np.float64, consumer=False)
    pipeline.clear_shms([name])
    _cleanup_shm(shm)


def test_hardware_launcher_write_and_read():
    hl = pipeline.hardwareLauncher("dummy.py", "c.yaml", 9999)
    calls = []

    def _write(msg):
        calls.append(msg)

    def _read():
        return {"status": "OK", "property": 42}

    hl.running = True
    hl.write = _write
    hl.read = _read
    assert hl.getProperty("x") == 42
    assert calls[0]["type"] == "get"
