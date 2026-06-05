"""Synthetic hardware components for onboarding and test flows.

The classes in this module emulate a minimal Shack-Hartmann sensor path and a
science camera without requiring external SDKs or laboratory hardware. They are
designed to exercise the normal pyRTC control pipeline rather than to model a
particular instrument with high optical fidelity.
"""

import time

import numpy as np

from pyRTC.Pipeline import initExistingShm
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.utils import setFromConfig


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

        self.frameRateHz = float(_numeric_from_config(conf, "frameRateHz", 200.0))
        self.framePeriod = 0.0 if self.frameRateHz <= 0 else 1.0 / self.frameRateHz
        self.backgroundLevel = float(_numeric_from_config(conf, "backgroundLevel", 150.0))
        self.spotFlux = float(_numeric_from_config(conf, "spotFlux", 3500.0))
        self.spotSigmaPx = float(_numeric_from_config(conf, "spotSigmaPx", 1.1))
        self.readNoise = float(_numeric_from_config(conf, "readNoise", 4.0))
        self.disturbanceAmplitude = float(_numeric_from_config(conf, "disturbanceAmplitude", 0.35))
        self.disturbanceFrequencyHz = float(_numeric_from_config(conf, "disturbanceFrequencyHz", 1.0))
        self.disturbanceDriftHz = float(_numeric_from_config(conf, "disturbanceDriftHz", 0.35))
        self.maxSpotMotionPx = float(_numeric_from_config(conf, "maxSpotMotionPx", 1.25))
        self.slopeToPixelGain = float(_numeric_from_config(conf, "slopeToPixelGain", 1.6))
        self.subApSpacing = int(setFromConfig(conf, "subApSpacing", 8))
        self.subApOffsetX = int(setFromConfig(conf, "subApOffsetX", 0))
        self.subApOffsetY = int(setFromConfig(conf, "subApOffsetY", 0))
        self.seed = int(setFromConfig(conf, "seed", 7))

        self.numRegions = min(
            (self.imageShape[0] - self.subApOffsetY) // self.subApSpacing,
            (self.imageShape[1] - self.subApOffsetX) // self.subApSpacing,
        )
        if self.numRegions < 1:
            raise ValueError("SyntheticSHWFS requires at least one valid SHWFS sub-aperture")

        self.signal2DShape = (2 * self.numRegions, self.numRegions)
        self.signalSize = int(np.prod(self.signal2DShape))
        self.numModes = int(setFromConfig(conf, "numModes", self.signalSize))
        if self.numModes < 1:
            raise ValueError("SyntheticSHWFS requires numModes >= 1")
        self.wfcLayout = _default_wfc_layout(self.numModes)

        self.rng = np.random.default_rng(self.seed)
        self.responseMatrix = self._build_response_matrix()
        self.modalPhases = self.rng.uniform(0.0, 2.0 * np.pi, self.numModes).astype(np.float32)
        self.modalAmplitudes = (
            self.disturbanceAmplitude / np.sqrt(np.arange(1, self.numModes + 1, dtype=np.float32))
        ).astype(np.float32)
        self.modalFrequencies = np.linspace(
            self.disturbanceFrequencyHz,
            self.disturbanceFrequencyHz + self.disturbanceDriftHz,
            self.numModes,
            dtype=np.float32,
        )
        self.localCoords = (np.arange(self.subApSpacing, dtype=np.float32) + 0.5) - (0.5 * self.subApSpacing)
        self.localGridX, self.localGridY = np.meshgrid(self.localCoords, self.localCoords)
        self.startTime = time.perf_counter()
        self.lastExposeTime = self.startTime
        self.frameCounter = 0
        self.lastModalDisturbance = np.zeros(self.numModes, dtype=np.float32)
        self.lastSlopeSignal = np.zeros(self.signalSize, dtype=np.float32)
        self.lastCorrection = np.zeros(self.numModes, dtype=np.float32)
        self.correctionShm = None

        return

    def _build_response_matrix(self):
        return build_synthetic_shwfs_response_matrix(self.numRegions, self.numModes, self.wfcLayout)

    def _sleep_for_frame_rate(self):
        if self.framePeriod <= 0.0:
            return
        elapsed = time.perf_counter() - self.lastExposeTime
        remaining = self.framePeriod - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _ensure_correction_stream(self):
        if self.correctionShm is not None:
            return
        try:
            self.correctionShm, _, _ = initExistingShm(self.input_stream_name("wfc"), gpuDevice=self.gpuDevice)
        except Exception:
            self.correctionShm = None

    def _modal_disturbance(self, elapsed_seconds):
        primary = np.sin(2.0 * np.pi * self.modalFrequencies * elapsed_seconds + self.modalPhases)
        secondary = np.cos(
            2.0 * np.pi * (0.37 * self.modalFrequencies + 0.11) * elapsed_seconds
            + 0.5 * self.modalPhases
        )
        disturbance = self.modalAmplitudes * (primary + 0.35 * secondary)
        return disturbance.astype(np.float32)

    def _current_correction(self):
        self._ensure_correction_stream()
        if self.correctionShm is None:
            return np.zeros(self.numModes, dtype=np.float32)

        correction = np.asarray(self.correctionShm.read_noblock(SAFE=False), dtype=np.float32).ravel()
        if correction.size < self.numModes:
            padded = np.zeros(self.numModes, dtype=np.float32)
            padded[: correction.size] = correction
            return padded
        return correction[: self.numModes].astype(np.float32, copy=False)

    def _render_spot_grid(self, slopes_2d):
        image = np.full(self.imageShape, self.backgroundLevel, dtype=np.float32)
        for row_index in range(self.numRegions):
            start_row = self.subApOffsetY + row_index * self.subApSpacing
            end_row = start_row + self.subApSpacing
            for col_index in range(self.numRegions):
                start_col = self.subApOffsetX + col_index * self.subApSpacing
                end_col = start_col + self.subApSpacing

                x_shift = np.clip(
                    self.slopeToPixelGain * slopes_2d[row_index, col_index],
                    -self.maxSpotMotionPx,
                    self.maxSpotMotionPx,
                )
                y_shift = np.clip(
                    self.slopeToPixelGain * slopes_2d[row_index + self.numRegions, col_index],
                    -self.maxSpotMotionPx,
                    self.maxSpotMotionPx,
                )
                patch = self.spotFlux * np.exp(
                    -(
                        (self.localGridX - x_shift) ** 2 + (self.localGridY - y_shift) ** 2
                    )
                    / (2.0 * self.spotSigmaPx**2)
                )
                image[start_row:end_row, start_col:end_col] += patch.astype(np.float32)

        if self.readNoise > 0.0:
            image += self.rng.normal(0.0, self.readNoise, size=self.imageShape).astype(np.float32)
        image = np.clip(image, 0.0, np.iinfo(self.imageRawDType).max)
        return image.astype(self.imageRawDType)

    def expose(self):
        self._sleep_for_frame_rate()
        elapsed_seconds = time.perf_counter() - self.startTime
        modal_disturbance = self._modal_disturbance(elapsed_seconds)
        correction = self._current_correction()
        slope_signal = self.responseMatrix @ (modal_disturbance - correction)
        slopes_2d = slope_signal.reshape(self.signal2DShape)

        self.lastModalDisturbance = modal_disturbance
        self.lastCorrection = correction
        self.lastSlopeSignal = slope_signal.astype(np.float32, copy=False)
        self.data = self._render_spot_grid(slopes_2d)
        self.frameCounter += 1
        self.lastExposeTime = time.perf_counter()
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

        self.frameRateHz = float(_numeric_from_config(conf, "frameRateHz", 50.0))
        self.framePeriod = 0.0 if self.frameRateHz <= 0 else 1.0 / self.frameRateHz
        self.backgroundLevel = float(_numeric_from_config(conf, "backgroundLevel", 50.0))
        self.peakFlux = float(_numeric_from_config(conf, "peakFlux", 25000.0))
        self.baseSigmaPx = float(_numeric_from_config(conf, "baseSigmaPx", 1.6))
        self.residualBlurGain = float(_numeric_from_config(conf, "residualBlurGain", 2.5))
        self.tipTiltGain = float(_numeric_from_config(conf, "tipTiltGain", 2.0))
        self.readNoise = float(_numeric_from_config(conf, "readNoise", 3.0))
        self.seed = int(setFromConfig(conf, "seed", 13))

        self.rng = np.random.default_rng(self.seed)
        self.signalShm = None
        self.frameCounter = 0
        self.lastExposeTime = time.perf_counter()
        self.gridY, self.gridX = np.indices(self.imageShape, dtype=np.float32)
        self.centerY = 0.5 * (self.imageShape[0] - 1)
        self.centerX = 0.5 * (self.imageShape[1] - 1)

        return

    def _sleep_for_frame_rate(self):
        if self.framePeriod <= 0.0:
            return
        elapsed = time.perf_counter() - self.lastExposeTime
        remaining = self.framePeriod - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _ensure_signal_stream(self):
        if self.signalShm is not None:
            return
        try:
            self.signalShm, _, _ = initExistingShm(self.input_stream_name("signal"))
        except Exception:
            self.signalShm = None

    def _current_signal(self):
        self._ensure_signal_stream()
        if self.signalShm is None:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(self.signalShm.read_noblock(SAFE=False), dtype=np.float32).ravel()

    def expose(self):
        self._sleep_for_frame_rate()
        signal = self._current_signal()
        residual_rms = float(np.sqrt(np.mean(signal**2))) if signal.size > 0 else 0.0

        tip = self.tipTiltGain * float(signal[0]) if signal.size > 0 else 0.0
        tilt_index = signal.size // 2
        tilt = self.tipTiltGain * float(signal[tilt_index]) if signal.size > 0 else 0.0
        sigma = self.baseSigmaPx + self.residualBlurGain * residual_rms
        peak = self.peakFlux / (1.0 + 2.0 * residual_rms)
        image = self.backgroundLevel + peak * np.exp(
            -(
                (self.gridX - (self.centerX + tip)) ** 2
                + (self.gridY - (self.centerY + tilt)) ** 2
            )
            / (2.0 * sigma**2)
        )
        if self.readNoise > 0.0:
            image += self.rng.normal(0.0, self.readNoise, size=self.imageShape)

        self.strehl_ratio = float(np.clip(1.0 / (1.0 + 3.0 * residual_rms), 0.0, 1.0))
        self.peak_dist = float(np.hypot(tip, tilt))
        self.strehlShm.write(np.array([self.strehl_ratio], dtype=float))
        self.tipTiltShm.write(np.array([self.peak_dist], dtype=float))
        self.data = np.clip(image, 0.0, np.iinfo(self.imageRawDType).max).astype(self.imageRawDType)
        self.frameCounter += 1
        self.lastExposeTime = time.perf_counter()
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
            self.setLayout(_default_wfc_layout(self.numActuators))


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