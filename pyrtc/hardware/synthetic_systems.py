"""Synthetic hardware components for onboarding and test flows.

The classes in this module emulate a minimal Shack-Hartmann sensor path and a
science camera without requiring external SDKs or laboratory hardware. They are
designed to exercise the normal pyrtc control pipeline rather than to model a
particular instrument with high optical fidelity.
"""

import time

import numpy as np

from pyrtc.streams import open_stream
from pyrtc.science_camera import ScienceCamera
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor
from pyrtc.utils import set_from_config


def _numeric_from_config(conf, key, default):
    """Read a numeric config value and fail fast on non-numeric input."""

    value = conf.get(key, default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return value


def _layout_sample_positions(layout: np.ndarray) -> np.ndarray:
    """Return normalized x/y sample positions for active boolean layout cells."""

    active_rows, active_cols = np.nonzero(layout)
    if active_rows.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    center_row = 0.5 * (layout.shape[0] - 1)
    center_col = 0.5 * (layout.shape[1] - 1)
    scale = max(center_row, center_col, 1.0)
    x = (active_cols.astype(np.float32) - center_col) / scale
    y = (active_rows.astype(np.float32) - center_row) / scale
    return np.column_stack((x, y)).astype(np.float32)


def build_synthetic_shwfs_response_matrix(num_regions: int, num_modes: int, layout: np.ndarray) -> np.ndarray:
    """Build a deterministic DM-to-slope response matrix for the synthetic AO example."""

    if num_regions < 1:
        raise ValueError("num_regions must be positive")
    if num_modes < 1:
        raise ValueError("num_modes must be positive")

    actuator_positions = _layout_sample_positions(layout)
    if actuator_positions.shape[0] < num_modes:
        raise ValueError(
            f"layout only exposes {actuator_positions.shape[0]} active actuator positions for {num_modes} modes"
        )
    actuator_positions = actuator_positions[:num_modes]

    subap_axis = np.linspace(-1.0, 1.0, num_regions, dtype=np.float32)
    subap_grid_x, subap_grid_y = np.meshgrid(subap_axis, subap_axis)
    subap_positions = np.column_stack((subap_grid_x.ravel(), subap_grid_y.ravel())).astype(np.float32)
    signal_size = 2 * subap_positions.shape[0]
    response = np.zeros((signal_size, num_modes), dtype=np.float32)
    influence_sigma = np.float32(0.38)

    for mode_index, actuator_position in enumerate(actuator_positions):
        delta = subap_positions - actuator_position
        radius_sq = np.sum(delta**2, axis=1)
        weight = np.exp(-0.5 * radius_sq / (influence_sigma**2)).astype(np.float32)
        response[: subap_positions.shape[0], mode_index] = delta[:, 0] * weight
        response[subap_positions.shape[0] :, mode_index] = delta[:, 1] * weight

    norms = np.linalg.norm(response, axis=0)
    norms[norms == 0.0] = 1.0
    response /= norms
    return response


class SyntheticSHWFS(WavefrontSensor):
    """
    Synthetic Shack-Hartmann wavefront sensor for no-hardware onboarding.

    The synthetic sensor reads the current correction vector from the standard
    ``wfc`` shared-memory stream, combines it with a deterministic modal
    disturbance, and renders a spot grid that can be consumed by the normal
    ``SlopesProcess`` SHWFS pipeline.
    """

    def __init__(self, conf):
        super().__init__(conf)

        self.frame_rate_hz = float(_numeric_from_config(conf, "frame_rate_hz", 200.0))
        self.frame_period = 0.0 if self.frame_rate_hz <= 0 else 1.0 / self.frame_rate_hz
        self.background_level = float(_numeric_from_config(conf, "background_level", 150.0))
        self.spot_flux = float(_numeric_from_config(conf, "spot_flux", 3500.0))
        self.spot_sigma_px = float(_numeric_from_config(conf, "spot_sigma_px", 1.1))
        self.read_noise = float(_numeric_from_config(conf, "read_noise", 4.0))
        self.disturbance_amplitude = float(_numeric_from_config(conf, "disturbance_amplitude", 0.35))
        self.disturbance_frequency_hz = float(_numeric_from_config(conf, "disturbance_frequency_hz", 1.0))
        self.disturbance_drift_hz = float(_numeric_from_config(conf, "disturbance_drift_hz", 0.35))
        self.max_spot_motion_px = float(_numeric_from_config(conf, "max_spot_motion_px", 1.25))
        self.slope_to_pixel_gain = float(_numeric_from_config(conf, "slope_to_pixel_gain", 1.6))
        self.sub_ap_spacing = int(set_from_config(conf, "sub_ap_spacing", 8))
        self.sub_ap_offset_x = int(set_from_config(conf, "sub_ap_offset_x", 0))
        self.sub_ap_offset_y = int(set_from_config(conf, "sub_ap_offset_y", 0))
        self.seed = int(set_from_config(conf, "seed", 7))

        self.num_regions = min(
            (self.image_shape[0] - self.sub_ap_offset_y) // self.sub_ap_spacing,
            (self.image_shape[1] - self.sub_ap_offset_x) // self.sub_ap_spacing,
        )
        if self.num_regions < 1:
            raise ValueError("SyntheticSHWFS requires at least one valid SHWFS sub-aperture")

        self.signal_2d_shape = (2 * self.num_regions, self.num_regions)
        self.signal_size = int(np.prod(self.signal_2d_shape))
        self.num_modes = int(set_from_config(conf, "num_modes", self.signal_size))
        if self.num_modes < 1:
            raise ValueError("SyntheticSHWFS requires num_modes >= 1")
        self.wfc_layout = _default_wfc_layout(self.num_modes)

        self.rng = np.random.default_rng(self.seed)
        self.response_matrix = self._build_response_matrix()
        self.modal_phases = self.rng.uniform(0.0, 2.0 * np.pi, self.num_modes).astype(np.float32)
        self.modal_amplitudes = (
            self.disturbance_amplitude / np.sqrt(np.arange(1, self.num_modes + 1, dtype=np.float32))
        ).astype(np.float32)
        self.modal_frequencies = np.linspace(
            self.disturbance_frequency_hz,
            self.disturbance_frequency_hz + self.disturbance_drift_hz,
            self.num_modes,
            dtype=np.float32,
        )
        self.local_coords = (np.arange(self.sub_ap_spacing, dtype=np.float32) + 0.5) - (0.5 * self.sub_ap_spacing)
        self.local_grid_x, self.local_grid_y = np.meshgrid(self.local_coords, self.local_coords)
        self.start_time = time.perf_counter()
        self.last_expose_time = self.start_time
        self.frame_counter = 0
        self.last_modal_disturbance = np.zeros(self.num_modes, dtype=np.float32)
        self.last_slope_signal = np.zeros(self.signal_size, dtype=np.float32)
        self.last_correction = np.zeros(self.num_modes, dtype=np.float32)
        self.correction_shm = None

        return

    def _build_response_matrix(self):
        return build_synthetic_shwfs_response_matrix(self.num_regions, self.num_modes, self.wfc_layout)

    def _sleep_for_frame_rate(self):
        if self.frame_period <= 0.0:
            return
        elapsed = time.perf_counter() - self.last_expose_time
        remaining = self.frame_period - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _ensure_correction_stream(self):
        if self.correction_shm is not None:
            return
        try:
            self.correction_shm = open_stream(self.input_stream_name("wfc"), gpu_device=self.gpu_device)
        except Exception:
            self.correction_shm = None

    def _modal_disturbance(self, elapsed_seconds):
        primary = np.sin(2.0 * np.pi * self.modal_frequencies * elapsed_seconds + self.modal_phases)
        secondary = np.cos(
            2.0 * np.pi * (0.37 * self.modal_frequencies + 0.11) * elapsed_seconds
            + 0.5 * self.modal_phases
        )
        disturbance = self.modal_amplitudes * (primary + 0.35 * secondary)
        return disturbance.astype(np.float32)

    def _current_correction(self):
        self._ensure_correction_stream()
        if self.correction_shm is None:
            return np.zeros(self.num_modes, dtype=np.float32)

        correction = np.asarray(self.correction_shm.read(), dtype=np.float32).ravel()
        if correction.size < self.num_modes:
            padded = np.zeros(self.num_modes, dtype=np.float32)
            padded[: correction.size] = correction
            return padded
        return correction[: self.num_modes].astype(np.float32, copy=False)

    def _render_spot_grid(self, slopes_2d):
        image = np.full(self.image_shape, self.background_level, dtype=np.float32)
        for row_index in range(self.num_regions):
            start_row = self.sub_ap_offset_y + row_index * self.sub_ap_spacing
            end_row = start_row + self.sub_ap_spacing
            for col_index in range(self.num_regions):
                start_col = self.sub_ap_offset_x + col_index * self.sub_ap_spacing
                end_col = start_col + self.sub_ap_spacing

                x_shift = np.clip(
                    self.slope_to_pixel_gain * slopes_2d[row_index, col_index],
                    -self.max_spot_motion_px,
                    self.max_spot_motion_px,
                )
                y_shift = np.clip(
                    self.slope_to_pixel_gain * slopes_2d[row_index + self.num_regions, col_index],
                    -self.max_spot_motion_px,
                    self.max_spot_motion_px,
                )
                patch = self.spot_flux * np.exp(
                    -(
                        (self.local_grid_x - x_shift) ** 2 + (self.local_grid_y - y_shift) ** 2
                    )
                    / (2.0 * self.spot_sigma_px**2)
                )
                image[start_row:end_row, start_col:end_col] += patch.astype(np.float32)

        if self.read_noise > 0.0:
            image += self.rng.normal(0.0, self.read_noise, size=self.image_shape).astype(np.float32)
        image = np.clip(image, 0.0, np.iinfo(self.image_raw_dtype).max)
        return image.astype(self.image_raw_dtype)

    def expose(self):
        self._sleep_for_frame_rate()
        elapsed_seconds = time.perf_counter() - self.start_time
        modal_disturbance = self._modal_disturbance(elapsed_seconds)
        correction = self._current_correction()
        slope_signal = self.response_matrix @ (modal_disturbance - correction)
        slopes_2d = slope_signal.reshape(self.signal_2d_shape)

        self.last_modal_disturbance = modal_disturbance
        self.last_correction = correction
        self.last_slope_signal = slope_signal.astype(np.float32, copy=False)
        self.data = self._render_spot_grid(slopes_2d)
        self.frame_counter += 1
        self.last_expose_time = time.perf_counter()
        super().expose()
        return


class SyntheticScienceCamera(ScienceCamera):
    """
    Synthetic PSF camera driven by the residual slope stream.

    This is intended for onboarding and viewer validation rather than optical
    fidelity. The PSF narrows and the synthetic Strehl rises as the residual
    signal norm falls.
    """

    def __init__(self, conf):
        super().__init__(conf)

        self.frame_rate_hz = float(_numeric_from_config(conf, "frame_rate_hz", 50.0))
        self.frame_period = 0.0 if self.frame_rate_hz <= 0 else 1.0 / self.frame_rate_hz
        self.background_level = float(_numeric_from_config(conf, "background_level", 50.0))
        self.peak_flux = float(_numeric_from_config(conf, "peak_flux", 25000.0))
        self.base_sigma_px = float(_numeric_from_config(conf, "base_sigma_px", 1.6))
        self.residual_blur_gain = float(_numeric_from_config(conf, "residual_blur_gain", 2.5))
        self.tip_tilt_gain = float(_numeric_from_config(conf, "tip_tilt_gain", 2.0))
        self.read_noise = float(_numeric_from_config(conf, "read_noise", 3.0))
        self.seed = int(set_from_config(conf, "seed", 13))

        self.rng = np.random.default_rng(self.seed)
        self.signal_shm = None
        self.frame_counter = 0
        self.last_expose_time = time.perf_counter()
        self.grid_y, self.grid_x = np.indices(self.image_shape, dtype=np.float32)
        self.center_y = 0.5 * (self.image_shape[0] - 1)
        self.center_x = 0.5 * (self.image_shape[1] - 1)

        return

    def _sleep_for_frame_rate(self):
        if self.frame_period <= 0.0:
            return
        elapsed = time.perf_counter() - self.last_expose_time
        remaining = self.frame_period - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _ensure_signal_stream(self):
        if self.signal_shm is not None:
            return
        try:
            self.signal_shm = open_stream(self.input_stream_name("signal"))
        except Exception:
            self.signal_shm = None

    def _current_signal(self):
        self._ensure_signal_stream()
        if self.signal_shm is None:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(self.signal_shm.read(), dtype=np.float32).ravel()

    def expose(self):
        self._sleep_for_frame_rate()
        signal = self._current_signal()
        residual_rms = float(np.sqrt(np.mean(signal**2))) if signal.size > 0 else 0.0

        tip = self.tip_tilt_gain * float(signal[0]) if signal.size > 0 else 0.0
        tilt_index = signal.size // 2
        tilt = self.tip_tilt_gain * float(signal[tilt_index]) if signal.size > 0 else 0.0
        sigma = self.base_sigma_px + self.residual_blur_gain * residual_rms
        peak = self.peak_flux / (1.0 + 2.0 * residual_rms)
        image = self.background_level + peak * np.exp(
            -(
                (self.grid_x - (self.center_x + tip)) ** 2
                + (self.grid_y - (self.center_y + tilt)) ** 2
            )
            / (2.0 * sigma**2)
        )
        if self.read_noise > 0.0:
            image += self.rng.normal(0.0, self.read_noise, size=self.image_shape)

        self.strehl_ratio = float(np.clip(1.0 / (1.0 + 3.0 * residual_rms), 0.0, 1.0))
        self.peak_dist = float(np.hypot(tip, tilt))
        self.strehl_shm.write(np.array([self.strehl_ratio], dtype=float))
        self.tip_tilt_shm.write(np.array([self.peak_dist], dtype=float))
        self.data = np.clip(image, 0.0, np.iinfo(self.image_raw_dtype).max).astype(self.image_raw_dtype)
        self.frame_counter += 1
        self.last_expose_time = time.perf_counter()
        super().expose()
        return


class SyntheticWFC(WavefrontCorrector):
    """Synthetic wavefront corrector used by onboarding and manager tests.

    The base ``WavefrontCorrector`` implementation already provides the behavior
    needed for a software-only control loop: it reads the modal correction
    stream, applies the M2C mapping, and publishes the optional 2D layout view.
    This subclass exists so configs can refer to a concrete synthetic adapter by
    name without implying vendor hardware.
    """
    def __init__(self, conf):
        super().__init__(conf)
        if self.layout is None:
            self.set_layout(_default_wfc_layout(self.num_actuators))

    @classmethod
    def sync_system_config(cls, system_conf):
        """Publish the synthetic layout size so SHM planning matches runtime.

        The generic ``wfc_2d`` planning spec assumes ``display_grid_size``; the
        synthetic corrector derives its layout from ``num_actuators``, so this
        hook keeps :func:`expected_output_shm_specs_for_config` consistent
        with the stream the component actually creates.
        """

        wfc_conf = system_conf.get("wfc")
        if isinstance(wfc_conf, dict) and "num_actuators" in wfc_conf:
            layout = _default_wfc_layout(int(wfc_conf["num_actuators"]))
            wfc_conf.setdefault("display_grid_size", int(layout.shape[0]))


def _default_wfc_layout(num_actuators: int) -> np.ndarray:
    """Return a centered, approximately circular boolean layout for synthetic DMs."""

    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")

    side = int(np.ceil(np.sqrt(float(num_actuators))))
    if side % 2 == 0:
        side += 1

    yy, xx = np.indices((side, side), dtype=np.float32)
    center = 0.5 * (side - 1)
    distances = (xx - center) ** 2 + (yy - center) ** 2
    selected = np.argsort(distances, axis=None)[:num_actuators]

    layout = np.zeros((side, side), dtype=bool)
    layout.flat[selected] = True
    return layout
