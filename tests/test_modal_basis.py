"""Tests for aobasis-backed modal bases (pyrtc.modal_basis and the WFC ``basis:`` section)."""

import importlib

import numpy as np
import pytest

from pyrtc.modal_basis import (
    BASIS_TYPES,
    actuator_positions_from_layout,
    build_m2c,
    default_actuator_layout,
    normalize_modes,
    parse_basis_config,
)

wfc_mod = importlib.import_module("pyrtc.wavefront_corrector")


def _alpao97_layout():
    yy, xx = np.mgrid[:11, :11]
    return np.sqrt((xx - 5) ** 2 + (yy - 5) ** 2) < 5.5


def _gram_offdiag_ratio(m2c):
    gram = m2c.T.astype(np.float64) @ m2c.astype(np.float64)
    off = gram - np.diag(np.diag(gram))
    return np.abs(off).max() / np.abs(np.diag(gram)).max()


# --------------------------------------------------------------------------
# Basis generation


@pytest.mark.parametrize("basis_type", BASIS_TYPES)
def test_every_basis_type_builds_full_rank_m2c(basis_type):
    layout = _alpao97_layout()
    num_modes = 4 if basis_type == "zonal_fast" else 40
    m2c = build_m2c(parse_basis_config({"type": basis_type}), num_modes=num_modes, layout=layout)

    assert m2c.shape == (97, num_modes)
    assert m2c.dtype == np.float32
    assert np.all(np.isfinite(m2c))
    assert np.linalg.matrix_rank(m2c) == num_modes
    # Default "peak" normalization: each mode's largest command is 1.
    np.testing.assert_allclose(np.abs(m2c).max(axis=0), 1.0, rtol=1e-6)


@pytest.mark.parametrize("basis_type", ["kl", "zonal", "zonal_fast"])
def test_natively_orthogonal_bases(basis_type):
    num_modes = 4 if basis_type == "zonal_fast" else 30
    m2c = build_m2c(
        parse_basis_config({"type": basis_type, "normalize": "none"}),
        num_modes=num_modes,
        layout=_alpao97_layout(),
    )
    assert _gram_offdiag_ratio(m2c) < 1e-5


@pytest.mark.parametrize("basis_type", BASIS_TYPES)
def test_orthonormalize_gives_orthonormal_columns(basis_type):
    num_modes = 4 if basis_type == "zonal_fast" else 30
    m2c = build_m2c(
        parse_basis_config({"type": basis_type, "orthonormalize": True, "normalize": "l2"}),
        num_modes=num_modes,
        layout=_alpao97_layout(),
    )
    np.testing.assert_allclose(m2c.T @ m2c, np.eye(num_modes), atol=1e-5)


def test_orthonormalize_keeps_mode_order_and_sign():
    layout = _alpao97_layout()
    raw = build_m2c(
        parse_basis_config({"type": "zernike", "normalize": "l2", "orthonormalize": False}),
        num_modes=5,
        layout=layout,
    )
    ortho = build_m2c(
        parse_basis_config({"type": "zernike", "normalize": "l2", "orthonormalize": True}),
        num_modes=5,
        layout=layout,
    )
    # The first mode is only rescaled; later modes stay positively correlated.
    np.testing.assert_allclose(ortho[:, 0], raw[:, 0], atol=1e-6)
    assert np.all(np.sum(ortho * raw, axis=0) > 0)


def test_ignore_piston_defaults_to_true():
    layout = _alpao97_layout()
    with_default = build_m2c(
        parse_basis_config({"type": "zernike", "normalize": "none", "orthonormalize": False}),
        num_modes=3,
        layout=layout,
    )
    with_piston = build_m2c(
        parse_basis_config(
            {"type": "zernike", "normalize": "none", "orthonormalize": False, "ignore_piston": False}
        ),
        num_modes=3,
        layout=layout,
    )
    np.testing.assert_allclose(with_piston[:, 0], 1.0)
    np.testing.assert_allclose(with_default[:, :2], with_piston[:, 1:3], atol=1e-6)


@pytest.mark.parametrize(
    "how, expected",
    [
        ("peak", lambda m: np.abs(m).max(axis=0)),
        ("rms", lambda m: np.sqrt(np.mean(m**2, axis=0))),
        ("l2", lambda m: np.linalg.norm(m, axis=0)),
    ],
)
def test_normalizations(how, expected):
    modes = np.array([[2.0, 0.0, -3.0], [1.0, 0.0, 4.0]])
    out = normalize_modes(modes, how)
    np.testing.assert_allclose(expected(out)[[0, 2]], 1.0)
    assert np.all(out[:, 1] == 0.0)  # zero columns are left alone


def test_too_many_modes_is_rejected():
    with pytest.raises(ValueError, match="num_modes must be <="):
        build_m2c(parse_basis_config({"type": "hadamard"}), num_modes=10, layout=np.ones((3, 3)))


def test_rank_deficient_basis_logs_warning(caplog):
    import logging

    logger = logging.getLogger("test_modal_basis")
    with caplog.at_level(logging.WARNING, logger="test_modal_basis"):
        build_m2c(
            parse_basis_config({"type": "zernike"}),
            num_modes=97,
            layout=_alpao97_layout(),
            logger=logger,
        )
    # Checked on the raw modes, so orthonormalizing (the Zernike default) does not hide it.
    assert "rank" in caplog.text
    assert "arbitrary orthogonal directions" in caplog.text


def test_fourier_without_piston_is_limited_to_one_fewer_mode_than_actuators():
    with pytest.raises(ValueError, match="maximum available is 96"):
        build_m2c(parse_basis_config({"type": "fourier"}), num_modes=97, layout=_alpao97_layout())


@pytest.mark.parametrize(
    ("basis_type", "expected"),
    [("zernike", True), ("fourier", True), ("kl", False), ("hadamard", False), ("zonal", False)],
)
def test_orthonormalize_default_depends_on_basis_type(basis_type, expected):
    assert parse_basis_config({"type": basis_type}).orthonormalize is expected
    assert parse_basis_config({"type": basis_type, "orthonormalize": not expected}).orthonormalize is (
        not expected
    )


@pytest.mark.parametrize("basis_type", ["zernike", "fourier"])
def test_default_zernike_and_fourier_bases_are_orthogonal(basis_type):
    m2c = build_m2c(
        parse_basis_config({"type": basis_type, "normalize": "l2"}),
        num_modes=50,
        layout=_alpao97_layout(),
    )
    np.testing.assert_allclose(m2c.T @ m2c, np.eye(50), atol=1e-5)


def test_kl_shape_depends_on_pupil_diameter_over_outer_scale():
    layout = _alpao97_layout()
    small = build_m2c(
        parse_basis_config({"type": "kl", "pupil_diameter": 0.1}), num_modes=10, layout=layout
    )
    large = build_m2c(
        parse_basis_config({"type": "kl", "pupil_diameter": 30.0}), num_modes=10, layout=layout
    )
    assert small.shape == large.shape
    assert not np.allclose(np.abs(small), np.abs(large), atol=1e-3)


# --------------------------------------------------------------------------
# Coordinates


def test_positions_from_layout_are_centred_row_major_metres():
    layout = np.zeros((3, 5), dtype=bool)
    layout[0, 0] = True
    layout[1, 2] = True
    layout[2, 4] = True
    positions = actuator_positions_from_layout(layout, pupil_diameter=8.0)

    # pitch = D / (max(shape) - 1) = 2 m; x follows columns, y rows.
    np.testing.assert_allclose(positions, [[-4.0, -2.0], [0.0, 0.0], [4.0, 2.0]])


def test_positions_file_overrides_layout(tmp_path):
    positions = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    path = tmp_path / "pos.npy"
    np.save(path, positions)
    m2c = build_m2c(
        parse_basis_config(
            {
                "type": "zernike",
                "positions_file": str(path),
                "normalize": "none",
                "orthonormalize": False,
            }
        ),
        num_modes=2,
        layout=np.ones((5, 5), dtype=bool),  # ignored
    )
    # Default pupil diameter is twice the largest actuator radius, so tip is x
    # times the Noll factor 2.
    np.testing.assert_allclose(m2c[:, 0], [-2.0, 0.0, 2.0, 0.0], atol=1e-6)


def test_default_layout_matches_actuator_count():
    for count in (1, 4, 50, 97):
        layout = default_actuator_layout(count)
        assert layout.shape[0] % 2 == 1
        assert int(layout.sum()) == count


# --------------------------------------------------------------------------
# Config parsing


def test_parse_basis_config_normalizes_values():
    basis = parse_basis_config(
        {"type": "KL", "n_modes": 10, "r0": 0.2, "L0": 25, "ignore_piston": False}, num_modes=10
    )
    assert basis.type == "kl"
    assert basis.n_modes == 10
    assert basis.normalize == "peak"
    assert basis.params == {"r0": 0.2, "L0": 25.0, "ignore_piston": False}
    assert parse_basis_config({"type": "zonal-fast"}).type == "zonal_fast"
    assert parse_basis_config(None) is None
    assert parse_basis_config({}) is None


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"type": "bessel"}, "basis.type must be one of"),
        ({}, None),
        ({"n_modes": 3}, "basis.type is required"),
        ({"type": "zonal", "r0": 0.1}, "unknown key"),
        ({"type": "kl", "r0": -1}, "r0 must be positive"),
        ({"type": "kl", "n_modes": 3}, "must equal num_modes"),
        ({"type": "kl", "normalize": "max"}, "basis.normalize"),
        ({"type": "kl", "orthonormalize": "yes"}, "orthonormalize"),
        ("kl", "must be a mapping"),
    ],
)
def test_parse_basis_config_rejects_bad_input(raw, match):
    if match is None:
        assert parse_basis_config(raw, num_modes=4) is None
        return
    with pytest.raises(ValueError, match=match):
        parse_basis_config(raw, num_modes=4)


def test_with_defaults_only_fills_unset_fields():
    basis = parse_basis_config({"type": "kl", "pupil_diameter": 2.0})
    assert basis.with_defaults(pupil_diameter=8.0).pupil_diameter == 2.0
    assert (
        parse_basis_config({"type": "kl"}).with_defaults(pupil_diameter=8.0).pupil_diameter == 8.0
    )


def test_wfc_config_validation_reports_basis_errors():
    from pyrtc.utils import ConfigValidationError, validate_wfc_config

    conf = {"name": "wfc", "num_actuators": 9, "num_modes": 4}
    validate_wfc_config({**conf, "basis": {"type": "zernike"}})
    with pytest.raises(ConfigValidationError, match="wfc: basis.type"):
        validate_wfc_config({**conf, "basis": {"type": "nope"}})


def test_descriptor_declares_basis_field():
    from pyrtc.component_descriptors import (
        get_component_descriptor,
        validate_config_with_descriptor,
    )

    field = get_component_descriptor("wfc").field_map["basis"]
    assert field.field_type == "dict | None"
    conf = {"name": "wfc", "num_actuators": 9, "num_modes": 4}
    validate_config_with_descriptor("wfc", {**conf, "basis": {"type": "kl"}})
    validate_config_with_descriptor("wfc", {**conf, "basis": None})
    with pytest.raises(TypeError, match="basis"):
        validate_config_with_descriptor("wfc", {**conf, "basis": "kl"})


# --------------------------------------------------------------------------
# WavefrontCorrector integration


def _wfc_conf(**extra):
    conf = {"name": "wfc", "num_actuators": 97, "num_modes": 20, "functions": []}
    conf.update(extra)
    return conf


def test_wavefront_corrector_builds_m2c_from_basis(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)

    wfc = wfc_mod.WavefrontCorrector(_wfc_conf(basis={"type": "kl", "n_modes": 20}))
    assert wfc.m2c_source == "basis"
    assert wfc.M2C.shape == (97, 20)
    assert wfc.C2M.shape == (20, 97)
    # Without a layout, the default circular layout for 97 actuators is used.
    expected = build_m2c(
        parse_basis_config({"type": "kl"}), num_modes=20, layout=default_actuator_layout(97)
    )
    np.testing.assert_allclose(wfc.M2C, expected, atol=1e-6)

    # The real layout rebuilds the basis on the true geometry.
    wfc.set_layout(_alpao97_layout())
    expected = build_m2c(parse_basis_config({"type": "kl"}), num_modes=20, layout=_alpao97_layout())
    np.testing.assert_allclose(wfc.M2C, expected, atol=1e-6)

    wfc.write(np.ones(20, dtype=np.float32))
    wfc.send_to_hardware()
    np.testing.assert_allclose(wfc.current_shape, wfc.M2C @ np.ones(20), atol=1e-5)

    # An explicit new basis replaces the configured one.
    wfc.build_basis_m2c({"type": "zonal"})
    np.testing.assert_allclose(wfc.M2C, np.eye(97)[:, :20])


def test_m2c_file_takes_precedence_over_basis(monkeypatch, tmp_path):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    m2c = np.random.default_rng(0).standard_normal((97, 20)).astype(np.float32)
    path = tmp_path / "m2c.npy"
    np.save(path, m2c)

    wfc = wfc_mod.WavefrontCorrector(_wfc_conf(m2c_file=str(path), basis={"type": "kl"}))
    assert wfc.m2c_source == "file"
    np.testing.assert_allclose(wfc.M2C, m2c)

    # A layout change must not replace a file-loaded M2C.
    wfc.set_layout(_alpao97_layout())
    np.testing.assert_allclose(wfc.M2C, m2c)


def test_wavefront_corrector_without_basis_keeps_identity(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = wfc_mod.WavefrontCorrector(_wfc_conf())
    assert wfc.m2c_source == "identity"
    np.testing.assert_allclose(wfc.M2C, np.eye(97)[:, :20])


def test_wavefront_corrector_rejects_layout_count_mismatch(monkeypatch):
    from testsupport import private_stream

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = wfc_mod.WavefrontCorrector(_wfc_conf(basis={"type": "zernike"}))
    with pytest.raises(ValueError, match="active actuators"):
        wfc.set_layout(np.ones((3, 3), dtype=bool))


def test_synthetic_wfc_uses_basis_on_its_layout(monkeypatch):
    from testsupport import private_stream

    from pyrtc.hardware import synthetic_systems

    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    wfc = synthetic_systems.SyntheticWFC(
        _wfc_conf(num_actuators=16, num_modes=6, basis={"type": "zernike"})
    )
    assert wfc.m2c_source == "basis"
    assert wfc.M2C.shape == (16, 6)
    expected = build_m2c(
        parse_basis_config({"type": "zernike"}),
        num_modes=6,
        layout=synthetic_systems._default_wfc_layout(16),
    )
    np.testing.assert_allclose(wfc.M2C, expected, atol=1e-6)


def test_gui_coerces_basis_mappings():
    from pyrtc.gui.manager_adapter import _coerce_runtime_value

    assert _coerce_runtime_value("{type: kl, r0: 0.2}", "dict | None") == {"type": "kl", "r0": 0.2}
    assert _coerce_runtime_value("", "dict | None") is None
    with pytest.raises(ValueError):
        _coerce_runtime_value("[1, 2]", "dict | None")
