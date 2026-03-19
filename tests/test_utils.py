import os
import io

import numpy as np
from astropy.io import fits

import pyRTC.utils as utils


def test_power_law_og_shape():
    arr = utils.powerLawOG(8, 2)
    assert arr.shape == (8,)
    assert arr[0] == 1


def test_append_to_file_and_roundtrip(tmp_path):
    file_path = tmp_path / "arr.bin"
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5], dtype=np.float32)
    utils.append_to_file(str(file_path), a)
    utils.append_to_file(str(file_path), b)
    out = np.fromfile(file_path, dtype=np.float32)
    assert np.array_equal(out, np.array([1, 2, 3, 4, 5], dtype=np.float32))


def test_generate_circular_aperture_mask():
    mask = utils.generate_circular_aperture_mask(20, 8, 0.25)
    assert mask.dtype == bool
    assert mask.shape == (20, 20)
    assert np.any(mask)


def test_load_data_npy_and_fits(tmp_path):
    arr = np.arange(9).reshape(3, 3)
    npy_file = tmp_path / "x.npy"
    fits_file = tmp_path / "x.fits"
    np.save(npy_file, arr)
    fits.PrimaryHDU(arr).writeto(fits_file)

    out_npy = utils.load_data(str(npy_file), dtype=np.float32)
    out_fits = utils.load_data(str(fits_file))
    assert out_npy.dtype == np.float32
    assert np.array_equal(out_fits, arr)


def test_generate_and_tmp_filepath(tmp_path):
    p = utils.generate_filepath(str(tmp_path), prefix="abc", extension=".dat")
    assert str(tmp_path) in p
    assert p.endswith(".dat")
    tmp = utils.get_tmp_filepath("/a/b/c.npy", uniqueStr="u")
    assert tmp.endswith("c_u.npy")


def test_centroid_and_buffer():
    arr = np.zeros((5, 5), dtype=float)
    arr[2, 4] = 3
    c = utils.centroid(arr)
    assert c.shape == (2,)
    buf = np.zeros((3, 2), dtype=float)
    utils.add_to_buffer(buf, np.array([7, 8]))
    assert np.array_equal(buf[-1], np.array([7, 8]))


def test_next_power_and_similarities():
    assert utils.next_power_of_two(0) == 1
    assert utils.next_power_of_two(5) == 8
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert np.isclose(utils.cosine_similarity(a, b), 1.0)
    assert np.isclose(utils.adjusted_cosine_similarity(a, b), 1.0)


def test_robust_variance_and_angle():
    x = np.array([1, 1, 2, 2, 100], dtype=float)
    assert utils.robust_variance(x) >= 0
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    ang = utils.angle_between_vectors(a, b)
    assert ang > 0


def test_image_cleanup_and_gaussian_grid():
    img = np.random.RandomState(0).randn(16, 16)
    out = utils.clean_image_for_strehl(img, median_filter_size=1, gaussian_sigma=0)
    assert out.shape == img.shape
    g = utils.gaussian_2d_grid(1, 1, sigma=1.0, grid_size=5)
    assert g.shape == (5, 5)
    assert np.isclose(np.sum(g), 1.0)
    assert np.isclose(g[1, 1], 0.0)


def test_set_affinity_invalid_type():
    assert utils.set_affinity("bad") == -1


def test_set_from_config_and_signal2d():
    conf = {"x": 2}
    assert utils.setFromConfig(conf, "x", 1) == 2
    assert utils.setFromConfig(conf, "x", 1.0) == 2.0
    layout = np.array([[True, False, True, False], [False, True, False, True]])
    signal = np.arange(np.count_nonzero(layout), dtype=float)
    out = utils.signal2D(signal, layout)
    assert out.shape == layout.shape


def test_dtype_roundtrip():
    idx = utils.dtype_to_float(np.float32)
    dt = utils.float_to_dtype(idx)
    assert dt == np.dtype(np.float32)


def test_precise_delay_and_change_path(tmp_path):
    utils.precise_delay(1)
    old = os.getcwd()
    utils.change_directory(str(tmp_path))
    assert os.getcwd() == str(tmp_path)
    os.chdir(old)


def test_measure_execution_time_and_add_to_path(tmp_path):
    calls = {"n": 0}

    def fn(a):
        calls["n"] += a

    med, iqr, c1, c99 = utils.measure_execution_time(fn, (1,), numIters=3)
    assert med >= 0
    assert iqr >= 0
    assert c1 <= c99

    utils.add_to_path(str(tmp_path))
    assert str(tmp_path) in os.environ.get("PATH", "")


def test_read_yaml_bind_socket_and_numeric(tmp_path, monkeypatch):
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text("a: 1\n")
    conf = utils.read_yaml_file(str(yaml_file))
    assert conf["a"] == 1

    sock = utils.bind_socket("127.0.0.1", 47000, max_attempts=2)
    assert sock is not None
    sock.close()

    monkeypatch.setattr(utils.select, "select", lambda *args, **kwargs: ([object()], [], []))
    monkeypatch.setattr(utils.sys, "stdin", io.StringIO("hello\n"))
    assert utils.read_input_with_timeout(0.01) == "hello"

    monkeypatch.setattr(utils.select, "select", lambda *args, **kwargs: ([], [], []))
    assert utils.read_input_with_timeout(0.01) is None

    assert utils.is_numeric("1.23")
    assert not utils.is_numeric("abc")


def test_load_data_unsupported(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("x")
    try:
        utils.load_data(str(bad))
        assert False
    except ValueError:
        assert True
