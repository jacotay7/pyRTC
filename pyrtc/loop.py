"""Adaptive optics loop control kernels and the main loop component.

This module contains the numerical update kernels and the high-level
``Loop`` component that turn measured residuals into new correction commands.
It is the control-plane heart of pyrtc: interaction matrices, control matrices,
integrators, and command dispatch all come together here.
"""

import math
import numpy as np
import time
from collections import deque
from typing import Any
from numba import jit
from scipy.linalg import hadamard

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import gpu_torch_available, open_stream
from pyrtc.component import Component
from pyrtc.utils import add_to_buffer, get_tmp_filepath, pyplot, set_from_config

logger = get_logger(__name__)

COMMON_CONDITIONING_LINES = (10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def leaky_integrator_numba(
    slopes: np.ndarray,
    reconstruction_matrix: np.ndarray,
    old_correction: np.ndarray,
    correction: np.ndarray,
    leak: np.float32,
    num_active_modes: int,
) -> np.ndarray:
    """Leaky-integrator update written into ``correction``.

    ``correction[:n] = (1 - leak) * old_correction[:n] - R[:n] @ slopes`` for
    the ``n = num_active_modes`` controlled modes; the remaining (dropped)
    modes are set to zero. ``correction`` is filled in place and returned.
    """
    update = np.dot(reconstruction_matrix[:num_active_modes], slopes)
    for i in range(num_active_modes):
        correction[i] = (1 - leak) * old_correction[i] - update[i]
    for i in range(num_active_modes, correction.size):
        correction[i] = 0.0
    return correction


def leak_integrator_gpu(
    slopes: np.ndarray,
    reconstruction_matrix: Any,
    old_correction: np.ndarray,
    leak: float,
    num_active_modes: int,
):
    """GPU counterpart of :func:`leaky_integrator_numba` (same semantics)."""

    if not gpu_torch_available():
        raise ImportError(
            "leak_integrator_gpu requires PyTorch. Install with 'pip install pyrtc[gpu]' or 'pip install torch'."
        )

    import torch

    slopes_gpu = torch.as_tensor(slopes, device=reconstruction_matrix.device)
    update = torch.matmul(reconstruction_matrix[:num_active_modes], slopes_gpu).cpu().numpy()
    correction = (1 - leak) * np.asarray(old_correction, dtype=update.dtype)
    correction[:num_active_modes] -= update
    correction[num_active_modes:] = 0
    return correction


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def comp_correction(cm=np.array([[]], dtype=np.float32), slopes=np.array([], dtype=np.float32)):
    """Apply a control matrix to a slope vector and return the correction."""

    return np.dot(cm, slopes)


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def update_correction(
    correction=np.array([], dtype=np.float32),
    g_cm=np.array([[]], dtype=np.float32),
    slopes=np.array([], dtype=np.float32),
):
    """Update an existing correction using a pre-scaled control matrix."""

    return correction - np.dot(g_cm, slopes)


# @jit(nopython=True)


class Loop(Component):
    """
    Real-time controller that closes the adaptive optics loop.

    ``Loop`` reads the current residual signal from the slopes pipeline,
    combines that signal with the calibrated control model, and writes the next
    correction vector to the wavefront-corrector stream. It also owns the
    operator-facing calibration state used to load or build interaction and
    control matrices and to tune classical integrator settings.

    In day-to-day use, this is the component that embodies the chosen control
    law for the system.

    Config
    ------
    num_dropped_modes : int, optional
        Number of modes to drop. Default is 0.
    gain : float, optional
        Gain for the integrator. Default is 0.1.
    leaky_gain : float, optional
        Leaky integrator gain. Default is 0.0.
    hardware_delay : float, optional
        Delay for the hardware. Default is 0.0.
    poke_amp : float, optional
        Amplitude for poking. Default is 0.01.
    num_iters_im : int, optional
        Number of iterations for interaction matrix computation. Default is 100.
    delay : int, optional
        Delay for corrections. Default is 0.
    im_method : str, optional
        Interaction-matrix calibration method: ``push-pull`` (one mode at a
        time), ``hadamard`` (multiplexed push-pull over Hadamard patterns) or
        ``docrime``. Default is "push-pull".
    im_settle_frames : int, optional
        Signal frames discarded after each calibration poke before averaging,
        so frames exposed while the corrector was still moving are not used.
        Default is 1.
    im_round_trip_check : bool, optional
        Run :meth:`check_round_trip` before :meth:`compute_im` calibrates, so
        calibration only starts once a DM command visibly reaches the signal.
        Default is True.
    im_timeout : float, optional
        Seconds :meth:`check_round_trip` may take, and the longest wait for
        any single signal frame during push-pull or Hadamard calibration.
        Default is 30.0.
    im_file : str, optional
        File to save the interaction matrix. Default is "".
    p_gain : float, optional
        Proportional gain for PID integrator. Default is 0.1.
    i_gain : float, optional
        Integral gain for PID integrator. Default is 0.0.
    d_gain : float, optional
        Derivative gain for PID integrator. Default is 0.0.
    control_limits : list, optional
        Control limits for PID integrator. Default is [-inf, inf].
    integral_limits : list, optional
        Integral limits for PID integrator. Default is [-inf, inf].
    absolute_limits : list, optional
        Absolute limits for corrections. Default is [-inf, inf].
    derivative_filter : float, optional
        Filter for the derivative term. Default is 0.1.

    Attributes
    ----------
    conf : dict
        Loop configuration.
    name : str
        Name of the loop.
    signal_dtype : type
        Data type of the wavefront sensor signal.
    signal_size : int
        Size of the wavefront sensor signal.
    signal_shm : pyshmem.SharedMemory
        Shared memory object for the wavefront sensor signal.
    null_signal : numpy.ndarray
        Null signal.
    signal_2d_dtype : type
        Data type of the 2D wavefront sensor signal.
    signal_2d_size : int
        Size of the 2D wavefront sensor signal.
    signal_2d_width : int
        Width of the 2D wavefront sensor signal.
    signal_2d_height : int
        Height of the 2D wavefront sensor signal.
    wfc_dtype : type
        Data type of the wavefront corrector.
    num_modes : int
        Number of modes in the wavefront corrector.
    wfc_shm : pyshmem.SharedMemory
        Shared memory object for the wavefront corrector.
    num_dropped_modes : int
        Number of dropped modes.
    num_active_modes : int
        Number of active modes.
    flat : numpy.ndarray
        Flat correction vector.
    im : numpy.ndarray
        Interaction matrix.
    cm : numpy.ndarray
        Control matrix.
    gain : float
        Gain for the integrator.
    leaky_gain : float
        Leaky integrator gain.
    perturb_amp : float
        Perturbation amplitude.
    hardware_delay : float
        Delay for the hardware.
    poke_amp : float
        Amplitude for poking.
    num_iters_im : int
        Number of iterations for interaction matrix computation.
    delay : int
        Delay for corrections.
    im_method : str
        Method for interaction matrix computation (lower case).
    im_settle_frames : int
        Frames discarded after each calibration poke.
    im_round_trip_check : bool
        Whether :meth:`compute_im` runs :meth:`check_round_trip` first.
    im_timeout : float
        Round-trip check budget and per-frame calibration timeout, in seconds.
    last_round_trip_frames : int or None
        Frame lag measured by the last :meth:`check_round_trip`.
    im_file : str
        File to save the interaction matrix.
    p_gain : float
        Proportional gain for PID integrator.
    i_gain : float
        Integral gain for PID integrator.
    d_gain : float
        Derivative gain for PID integrator.
    control_limits : list
        Control limits for PID integrator.
    integral_limits : list
        Integral limits for PID integrator.
    absolute_limits : list
        Absolute limits for corrections.
    derivative_filter : float
        Filter for the derivative term.
    cm_method : str
        Control-matrix inversion method. Supported values are ``svd`` and
        ``tikhonov``.
    conditioning : float or None
        Optional target conditioning number used to truncate small singular
        values when computing the control matrix.
    tikhonov_reg : float
        Tikhonov regularization strength used when ``cm_method`` is
        ``tikhonov``.
    integral : numpy.ndarray
        Integral term for PID integrator.
    previous_wf_error : numpy.ndarray
        Previous wavefront error.
    previous_derivative : numpy.ndarray
        Previous derivative term.
    control_output : numpy.ndarray
        Control output.
    """

    SUPPORTED_IM_METHODS = ("push-pull", "hadamard", "docrime")
    # Frames that must agree before check_round_trip treats the signal as settled.
    _ROUND_TRIP_WINDOW = 5

    def __init__(self, conf) -> None:
        """
        Constructs all the necessary attributes for the Loop object.

        Parameters
        ----------
        conf : dict
            Configuration dictionary with the following keys
            wfs : dict
                Wavefront sensor configuration.
            wfc : dict
                Wavefront corrector configuration.
            loop : dict
                Loop configuration containing
                num_dropped_modes : int, optional
                    Number of modes to drop. Default is 0.
                gain : float, optional
                    Gain for the integrator. Default is 0.1.
                leaky_gain : float, optional
                    Leaky integrator gain. Default is 0.0.
                hardware_delay : float, optional
                    Delay for the hardware. Default is 0.0.
                poke_amp : float, optional
                    Amplitude for poking. Default is 0.01.
                num_iters_im : int, optional
                    Number of iterations for interaction matrix computation. Default is 100.
                delay : int, optional
                    Delay for corrections. Default is 0.
                im_method : str, optional
                    Method for interaction matrix computation. Default is "push-pull".
                im_file : str, optional
                    File to save the interaction matrix. Default is "".
                p_gain : float, optional
                    Proportional gain for PID integrator. Default is 0.1.
                i_gain : float, optional
                    Integral gain for PID integrator. Default is 0.0.
                d_gain : float, optional
                    Derivative gain for PID integrator. Default is 0.0.
                control_limits : list, optional
                    Control limits for PID integrator. Default is [-inf, inf].
                integral_limits : list, optional
                    Integral limits for PID integrator. Default is [-inf, inf].
                absolute_limits : list, optional
                    Absolute limits for corrections. Default is [-inf, inf].
                derivative_filter : float, optional
                    Filter for the derivative term. Default is 0.1.
        """
        try:
            super().__init__(conf)
            self.name = "Loop"
            self.conf = conf

            # Read wfs signal's metadata and open a stream to the shared memory
            self.signal_shm = open_stream(
                self.input_stream_name("signal"), gpu_device=self.gpu_device
            )
            self.signal_shape = tuple(self.signal_shm.shape)
            self.signal_dtype = np.dtype(self.signal_shm.dtype)
            self.register_input_stream("signal", self.signal_shm)
            self.signal_size = int(np.prod(self.signal_shape))
            self.null_signal = np.zeros(self.signal_shape, dtype=self.signal_dtype)

            # Read wfc metadata and open a stream to the shared memory
            self.wfc_shm = open_stream(self.output_stream_name("wfc"), gpu_device=self.gpu_device)
            self.wfc_shape = tuple(self.wfc_shm.shape)
            self.wfc_dtype = np.dtype(self.wfc_shm.dtype)
            self.register_output_stream("wfc", self.wfc_shm)
            self.num_modes = int(np.prod(self.wfc_shape))

            self.num_dropped_modes = set_from_config(self.conf, "num_dropped_modes", 0)
            self.num_active_modes = self.num_modes - self.num_dropped_modes
            self.flat = np.zeros(self.num_modes, dtype=self.wfc_dtype)
            self.null_correction = np.zeros_like(self.flat)
            self._correction_buffer = np.zeros_like(self.flat)

            self.im = np.zeros((self.signal_size, self.num_modes), dtype=self.signal_dtype)
            self.cm = np.zeros((self.num_modes, self.signal_size), dtype=self.signal_dtype)
            self.gain = set_from_config(self.conf, "gain", 0.1)
            self.leaky_gain = set_from_config(self.conf, "leaky_gain", 0.0)
            self.perturb_amp = 0
            self.hardware_delay = set_from_config(self.conf, "hardware_delay", 0.0)
            self.poke_amp = set_from_config(self.conf, "poke_amp", 1e-2)
            self.num_iters_im = set_from_config(self.conf, "num_iters_im", 100)
            self.delay = set_from_config(self.conf, "delay", 0)
            self.im_method = self._validate_im_method(
                set_from_config(self.conf, "im_method", "push-pull")
            )
            self.im_settle_frames = int(set_from_config(self.conf, "im_settle_frames", 1))
            if self.im_settle_frames < 0:
                raise ValueError("im_settle_frames must be >= 0")
            self.im_round_trip_check = bool(set_from_config(self.conf, "im_round_trip_check", True))
            self.im_timeout = float(set_from_config(self.conf, "im_timeout", 30.0))
            self.last_round_trip_frames = None
            self.im_file = set_from_config(self.conf, "im_file", "")
            self.cm_method = str(set_from_config(self.conf, "cm_method", "svd")).lower()
            conditioning = set_from_config(self.conf, "conditioning", None)
            self.conditioning = None if conditioning is None else float(conditioning)
            self.tikhonov_reg = float(set_from_config(self.conf, "tikhonov_reg", 0.0))
            self.last_singular_values = np.array([], dtype=np.float64)
            self.last_retained_singular_mask = np.array([], dtype=bool)
            self.last_suggested_conditioning = None
            self.last_singular_value_fit = None

            self.cl_docrime = False
            self.num_iters_dc = 0
            tmp2 = self.flat.copy().reshape(self.flat.size, 1)
            tmp = self.null_signal.copy().reshape(self.null_signal.size, 1)
            self.docrime_cross = np.zeros_like(tmp @ tmp2.T)
            self.docrime_auto = np.zeros_like(tmp2 @ tmp2.T)
            self.docrime_buffer = np.zeros((1 + self.delay, *tmp2.shape), dtype=self.wfc_dtype)

            self.p_gain = set_from_config(self.conf, "p_gain", 0.1)
            self.i_gain = set_from_config(self.conf, "i_gain", 0.0)
            self.d_gain = set_from_config(self.conf, "d_gain", 0.0)
            self.control_limits = set_from_config(self.conf, "control_limits", [-np.inf, np.inf])
            self.integral_limits = set_from_config(self.conf, "integral_limits", [-np.inf, np.inf])
            self.absolute_limits = set_from_config(self.conf, "absolute_limits", [-np.inf, np.inf])
            self.derivative_filter = set_from_config(self.conf, "derivative_filter", 0.1)
            self.integral = 0

            self.previous_wf_error = np.zeros_like(self.read_stream("wfc", block=False))
            self.previous_derivative = np.zeros_like(self.previous_wf_error)
            self.control_output = np.zeros_like(self.previous_wf_error)

            # Pre-allocated hot-path read buffers (ignored for GPU streams).
            self._signal_buffer = np.empty(self.signal_shape, dtype=self.signal_dtype)
            self._wfc_buffer = np.empty(self.wfc_shape, dtype=self.wfc_dtype)

            self.load_im()
            self.logger.info(
                "Initialized loop signal_shape=%s wfc_shape=%s num_modes=%s",
                self.signal_shape,
                self.wfc_shape,
                self.num_modes,
            )
        except Exception:
            logger.exception("Failed to initialize loop")
            raise

        return

    @property
    def gain(self):
        return getattr(self, "_gain", 0.0)

    @gain.setter
    def gain(self, gain):
        self._gain = float(gain)
        if hasattr(self, "cm"):
            self.g_cm = self._gain * self.cm

    def set_gain(self, gain):
        """
        Set the integrator gain. Only needed for certain integrators.

        Parameters
        ----------
        gain : float
            Gain to set.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.gain = gain
            component_logger.info("Set loop gain to %s", gain)
        except Exception:
            component_logger.exception("Failed to set loop gain to %s", gain)
            raise
        return

    def set_peturb_amp(self, amp):
        """
        Set the perturbation amplitude.

        Parameters
        ----------
        amp : float
            Amplitude to set.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.perturb_amp = amp
            component_logger.info("Set perturbation amplitude to %s", amp)
        except Exception:
            component_logger.exception("Failed to set perturbation amplitude to %s", amp)
            raise
        return

    @classmethod
    def _validate_im_method(cls, method) -> str:
        normalized = str(method).lower()
        if normalized not in cls.SUPPORTED_IM_METHODS:
            raise ValueError(
                f"Unsupported interaction-matrix method {method!r}; "
                f"expected one of {cls.SUPPORTED_IM_METHODS}"
            )
        return normalized

    def _next_signal(self, deadline=None, what="a new signal frame"):
        """Consume the next signal frame as a flat float64 vector.

        Waits at most until ``deadline`` (a ``time.monotonic()`` value), or
        :attr:`im_timeout` seconds when no deadline is given, and raises
        ``TimeoutError`` naming ``what`` was awaited.
        """

        if deadline is None:
            timeout = self.im_timeout
        else:
            timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError(f"timed out waiting for {what}")
        try:
            frame = self.read_stream("signal", timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"timed out after {timeout:.1f}s waiting for {what}") from exc
        if hasattr(frame, "detach"):  # GPU-attached signal stream (torch tensor)
            frame = frame.detach().cpu().numpy()
        return np.asarray(frame, dtype=np.float64).ravel()

    @staticmethod
    def _rms(vector) -> float:
        return float(np.sqrt(np.mean(np.square(vector)))) if vector.size else 0.0

    def _wait_for_stable_signal(self, deadline, what):
        """Read until the last few signal frames agree; return their mean and noise.

        A window of frames is *stable* when no frame strays from the window
        mean by more than three times the median frame-to-frame change. The
        test is relative to the sensor's own noise, so it works for noisy
        sensors and for noise-free simulations alike, and rejects a window
        that contains a step (for example a DM command landing).
        """

        frames = deque(maxlen=self._ROUND_TRIP_WINDOW)
        while True:
            frames.append(self._next_signal(deadline, what))
            if len(frames) < frames.maxlen:
                continue
            stack = np.stack(frames)
            mean = stack.mean(axis=0)
            noise = float(np.median([self._rms(b - a) for a, b in zip(stack[:-1], stack[1:])]))
            spread = max(self._rms(frame - mean) for frame in stack)
            if spread <= 3.0 * noise + 1e-9 * self._rms(mean):
                return mean, noise

    def _round_trip_pattern(self) -> np.ndarray:
        """Fixed +/-``poke_amp`` pattern over every mode used by :meth:`check_round_trip`."""

        signs = np.random.default_rng(87).choice((-1.0, 1.0), size=self.num_modes)
        return (self.flat + self.poke_amp * signs).astype(self.wfc_dtype)

    def check_round_trip(self, timeout=None):
        """Verify that a corrector command reaches the measured signal.

        Worker kernels JIT-compile on first use, so right after start-up the
        first DM command can take about a second to reach the signal stream.
        Calibrating in that window records zero or smeared IM columns. This
        check flattens the corrector, waits for stable frames, pokes every mode
        by +/-``poke_amp`` (a fixed sign pattern, the same per-mode amplitude
        a Hadamard calibration uses), waits for the signal to move and settle,
        flattens again, and waits for the signal to return. Transient frames
        (such as a stale frame from before the pipeline started) make it retry
        the cycle until ``timeout``.

        :meth:`compute_im` runs this first when ``im_round_trip_check`` is
        true (the default). Call it directly before steps that need the live
        pipeline but do not poke, such as taking reference slopes.

        Parameters
        ----------
        timeout : float, optional
            Seconds to allow in total. Defaults to ``im_timeout``.

        Returns
        -------
        int
            Frames read after the flattening command up to and including the
            first one that shows it (1 means the next frame already did). Also
            stored on :attr:`last_round_trip_frames`.

        Raises
        ------
        TimeoutError
            If the signal never responds to the poke (``poke_amp`` too small to
            measure, corrector or sensor not running), never settles, or never
            returns to the flat-DM signal within ``timeout``.
        """

        component_logger = getattr(self, "logger", logger)
        timeout = self.im_timeout if timeout is None else float(timeout)
        if self.poke_amp <= 0:
            raise ValueError("poke_amp must be positive to check the DM round trip")
        deadline = time.monotonic() + timeout
        poke = self._round_trip_pattern()

        try:
            self.flatten()
            # The first blocking read can return a payload published before the
            # flat command (or before the producer ever ran); drop it.
            self._next_signal(deadline, "a signal frame (is the WFS/slopes pipeline running?)")
            attempt = 0
            while True:
                attempt += 1
                baseline, noise = self._wait_for_stable_signal(
                    deadline, "the signal to settle on the flat DM"
                )
                threshold = max(3.0 * noise, 1e-6 * self._rms(baseline), 1e-12)

                self.send_to_wfc(poke)
                response_frames = 0
                while True:
                    signal = self._next_signal(
                        deadline,
                        f"the signal to respond to a DM poke of +/-{self.poke_amp} on every mode "
                        "(is poke_amp large enough to measure, and are the corrector and "
                        "sensor running?)",
                    )
                    response_frames += 1
                    if self._rms(signal - baseline) <= threshold:
                        continue
                    poked, _ = self._wait_for_stable_signal(
                        deadline, "the signal to settle on the poked DM"
                    )
                    # A single noisy frame is not a response; the settled level must move.
                    if self._rms(poked - baseline) > threshold:
                        break
                direction = poked - baseline
                amplitude_sq = float(np.dot(direction, direction))

                self.flatten()
                limit = max(50, 2 * response_frames)
                for frames in range(1, limit + 1):
                    signal = self._next_signal(
                        deadline, "the signal to return to its flat-DM value"
                    )
                    # Closer to the flat level than to the poked level.
                    if np.dot(signal - baseline, direction) < 0.5 * amplitude_sq:
                        # Leave the pipeline settled for whatever reads next
                        # (reference slopes, calibration).
                        self._wait_for_stable_signal(
                            deadline, "the signal to settle back on the flat DM"
                        )
                        self.last_round_trip_frames = frames
                        component_logger.info(
                            "DM round trip confirmed: poke seen after %s frame(s), flat after %s "
                            "frame(s) (attempt %s)",
                            response_frames,
                            frames,
                            attempt,
                        )
                        return frames
                component_logger.warning(
                    "Signal did not return to its flat-DM value within %s frames after a DM "
                    "round-trip poke; retrying",
                    limit,
                )
        except TimeoutError as exc:
            raise TimeoutError(
                f"DM round-trip check did not complete within {timeout:.1f}s: {exc}"
            ) from exc

    def _average_signal(self, correction):
        """Send ``correction``, drop ``im_settle_frames`` frames, and average ``num_iters_im``."""

        self.send_to_wfc(np.asarray(correction, dtype=self.wfc_dtype))
        if self.hardware_delay > 0:
            time.sleep(self.hardware_delay)
        settle_frames = getattr(self, "_active_settle_frames", None)
        if settle_frames is None:
            settle_frames = self.im_settle_frames
        for _ in range(settle_frames):
            self._next_signal(what="a calibration frame")
        total = np.zeros(self.signal_size, dtype=np.float64)
        for _ in range(self.num_iters_im):
            total += self._next_signal(what="a calibration frame")
        return total / self.num_iters_im

    def _push_pull_response(self, pattern):
        """Signal response per unit command to ``pattern`` (push minus pull)."""

        delta = self.poke_amp * np.asarray(pattern, dtype=np.float64)
        plus = self._average_signal(self.flat + delta)
        minus = self._average_signal(self.flat - delta)
        return (plus - minus) / (2.0 * self.poke_amp)

    def push_pull_im(self):
        """
        Compute the interaction matrix using the push-pull method.

        Each mode is poked to ``+poke_amp`` and ``-poke_amp`` in turn. After
        each poke ``im_settle_frames`` frames are discarded and ``num_iters_im``
        frames are averaged; the IM column is the difference over
        ``2 * poke_amp``.
        """
        if self.poke_amp <= 0:
            raise ValueError("poke_amp must be positive for push-pull calibration")
        im = np.zeros((self.signal_size, self.num_modes), dtype=np.float64)
        for i in range(self.num_modes):
            pattern = np.zeros(self.num_modes, dtype=np.float64)
            pattern[i] = 1.0
            im[:, i] = self._push_pull_response(pattern)
        self.im = im.astype(self.signal_dtype)
        return

    @staticmethod
    def hadamard_patterns(num_modes: int) -> np.ndarray:
        """Return the +/-1 poke patterns used by :meth:`hadamard_im`.

        The patterns are the rows of the Sylvester Hadamard matrix of order
        ``N``, the next power of two at or above ``num_modes``, truncated to
        the first ``num_modes`` columns. The result has shape
        ``(N, num_modes)`` and orthogonal columns (``P.T @ P == N * I``), so
        responses to the patterns demultiplex exactly into per-mode columns.
        """

        num_modes = int(num_modes)
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        order = 1 << (num_modes - 1).bit_length()
        return hadamard(order).astype(np.float64)[:, :num_modes]

    def hadamard_im(self):
        """
        Compute the interaction matrix with multiplexed (Hadamard) push-pull.

        Every mode is poked at once with the +/-``poke_amp`` sign patterns
        from :meth:`hadamard_patterns` (``N`` patterns, ``N`` the next power of
        two at or above the number of modes), each measured with push-pull
        like :meth:`push_pull_im`. The pattern responses ``S`` are
        demultiplexed as ``IM = S @ P / N``. For the same number of frames
        this averages each IM column over every measurement, cutting white
        measurement noise by about ``sqrt(num_modes)`` compared with
        push-pull. Every pattern moves all modes, so choose ``poke_amp`` small
        enough that the sensor stays linear for the combined shape.
        """
        if self.poke_amp <= 0:
            raise ValueError("poke_amp must be positive for Hadamard calibration")
        patterns = self.hadamard_patterns(self.num_modes)
        responses = np.zeros((self.signal_size, patterns.shape[0]), dtype=np.float64)
        for k, pattern in enumerate(patterns):
            responses[:, k] = self._push_pull_response(pattern)
        im = responses @ patterns / patterns.shape[0]
        self.im = im.astype(self.signal_dtype)
        return

    def docrime_im(self):
        """
        Compute the interaction matrix using the DOCRIME method.
        """
        # Send the flat command to the WFC
        self.flatten()

        # Get a correction to set the shape
        correction = self.flat.copy()
        correction = correction.reshape(correction.size, 1)

        # Have a history of corrections
        # corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        # Get an initial slope reading to set shapes
        slopes = self.null_signal.copy()
        slopes = slopes.reshape(slopes.size, 1)
        self.docrime_cross = np.zeros_like(self.docrime_cross)
        self.docrime_auto = np.zeros_like(self.docrime_auto)

        for i in range(self.num_iters_im):
            # Compute new random shape
            correction = (
                np.random.uniform(-self.poke_amp, self.poke_amp, correction.size)
                .astype(correction.dtype)
                .reshape(correction.shape)
            )

            # Get current WFS response
            # I put this first to match CL case
            slopes = self.read_stream("signal").reshape(slopes.shape)

            # Send random shape to mirror
            self.send_to_wfc(correction)

            add_to_buffer(self.docrime_buffer, correction)

            # Correlate Current response with old correction by delay time
            self.docrime_cross += slopes @ self.docrime_buffer[0].T
            self.docrime_auto += self.docrime_buffer[0] @ self.docrime_buffer[0].T

        self.docrime_cross /= self.num_iters_im
        self.docrime_auto /= self.num_iters_im
        self.im = self.docrime_cross @ np.linalg.inv(self.docrime_auto)

        self.docrime_cross = np.zeros_like(self.docrime_cross)
        self.docrime_auto = np.zeros_like(self.docrime_auto)

        return

    def compute_im(self, round_trip_check=None):
        """
        Compute the interaction matrix using the configured ``im_method``.

        ``push-pull`` (default) pokes one mode at a time, ``hadamard`` pokes
        Hadamard patterns over all modes and demultiplexes, and ``docrime``
        correlates random commands with the signal. Unless disabled,
        :meth:`check_round_trip` runs first so calibration only starts once
        the pipeline is live. The corrector is flattened afterwards and the
        control matrix is recomputed.

        Parameters
        ----------
        round_trip_check : bool, optional
            Override ``im_round_trip_check`` for this call.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            method = self._validate_im_method(self.im_method)
            check = self.im_round_trip_check if round_trip_check is None else round_trip_check
            settle_frames = self.im_settle_frames
            if check:
                frames = self.check_round_trip()
                # Discard at least as many frames as the measured round trip
                # took: on a slow or loaded pipeline a poke can take several
                # frames to land, and averaging earlier frames corrupts the IM.
                if method != "docrime" and frames > settle_frames:
                    component_logger.info(
                        "DM round trip took %s frames; discarding %s frames after each "
                        "calibration poke (im_settle_frames=%s)",
                        frames,
                        frames,
                        self.im_settle_frames,
                    )
                    settle_frames = frames
            self._active_settle_frames = settle_frames
            component_logger.info("Computing interaction matrix using method=%s", method)
            try:
                if method == "docrime":
                    self.docrime_im()
                elif method == "hadamard":
                    self.hadamard_im()
                else:
                    self.push_pull_im()
            finally:
                self._active_settle_frames = None
                try:
                    self.flatten()
                except Exception:
                    component_logger.exception("Failed to flatten after calibration")

            self.compute_cm()
        except Exception:
            component_logger.exception(
                "Failed to compute interaction matrix using method=%s",
                getattr(self, "im_method", None),
            )
            raise
        return

    def save_im(self, filename=""):
        """
        Save the interaction matrix to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the interaction matrix to. If not specified, uses the configured im_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.im_file
            if filename == "":
                raise ValueError("No interaction matrix filename provided")
            np.save(filename, self.im)
            component_logger.info("Saved interaction matrix to %s", filename)
        except Exception:
            component_logger.exception(
                "Failed to save interaction matrix to %s", filename or getattr(self, "im_file", "")
            )
            raise

    def load_im(self, filename=""):
        """
        Load the interaction matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the interaction matrix from. If not specified, uses the configured im_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.im_file
            if filename == "":
                self.im = np.zeros_like(self.im)
                component_logger.info("No interaction matrix file configured; using zeros")
            else:
                self.im = np.load(filename)
                component_logger.info("Loaded interaction matrix from %s", filename)
            self.compute_cm()
        except Exception:
            component_logger.exception(
                "Failed to load interaction matrix from %s",
                filename or getattr(self, "im_file", ""),
            )
            raise

    def flatten(self):
        """
        Send the flat correction to the wavefront corrector.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.send_to_wfc(self.flat)
            component_logger.info("Flattened loop correction")
        except Exception:
            component_logger.exception("Failed to flatten loop correction")
            raise
        return

    @staticmethod
    def _validate_cm_method(method: str) -> str:
        normalized = str(method).lower()
        if normalized not in {"svd", "tikhonov"}:
            raise ValueError(f"Unsupported CM inversion method: {method}")
        return normalized

    @staticmethod
    def _suggest_conditioning_from_singular_values(singular_values: np.ndarray):
        singular_values = np.asarray(singular_values, dtype=np.float64)
        singular_values = singular_values[np.isfinite(singular_values) & (singular_values > 0)]
        if singular_values.size < 4:
            return None, None

        normalized = singular_values / singular_values[0]
        indices = np.arange(normalized.size, dtype=np.float64)
        log_values = np.log10(np.clip(normalized, np.finfo(np.float64).tiny, None))

        min_leading_points = max(3, normalized.size // 8)
        best_score = -np.inf
        best_fit = None

        for knee_index in range(min_leading_points - 1, normalized.size - 1):
            leading_x = indices[: knee_index + 1]
            leading_y = log_values[: knee_index + 1]

            sample_count = float(leading_x.size)
            x_mean = math.fsum(float(value) for value in leading_x) / sample_count
            y_mean = math.fsum(float(value) for value in leading_y) / sample_count
            centered_x = leading_x - x_mean
            centered_y = leading_y - y_mean
            variance_x = float(np.dot(centered_x, centered_x))
            if variance_x <= 0:
                continue

            slope = float(np.dot(centered_x, centered_y) / variance_x)
            intercept = float(y_mean - slope * x_mean)
            fit_y = slope * leading_x + intercept
            fit_residual = leading_y - fit_y
            rmse = math.sqrt(
                math.fsum(float(value) * float(value) for value in fit_residual) / sample_count
            )

            predicted_next = slope * indices[knee_index + 1] + intercept
            downward_departure = predicted_next - log_values[knee_index + 1]
            if downward_departure <= 0:
                continue

            score = downward_departure / (rmse + 1e-6)
            if score > best_score:
                threshold = normalized[knee_index + 1]
                if threshold <= 0:
                    continue
                best_score = score
                best_fit = {
                    "knee_index": int(knee_index),
                    "suggested_index": int(knee_index + 1),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "rmse": rmse,
                    "normalized_threshold": float(threshold),
                    "conditioning": float(1.0 / threshold),
                    "score": float(score),
                    "indices": indices.copy(),
                    "normalized_singular_values": normalized.copy(),
                    "fit_curve": np.power(10.0, slope * indices + intercept),
                }

        if best_fit is None:
            return None, None
        return best_fit["conditioning"], best_fit

    def get_singular_values(self) -> np.ndarray:
        if self.im.size == 0:
            return np.array([], dtype=np.float64)
        return np.linalg.svd(self.im, compute_uv=False)

    def suggest_conditioning_number(self):
        singular_values = self.get_singular_values()
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.last_suggested_conditioning = suggestion
        self.last_singular_value_fit = fit
        return suggestion

    def plot_singular_values(self, conditioning_lines=COMMON_CONDITIONING_LINES, ax=None):
        singular_values = self.get_singular_values()
        self.last_singular_values = singular_values
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.last_suggested_conditioning = suggestion
        self.last_singular_value_fit = fit

        if ax is None:
            fig = pyplot().figure(figsize=(8, 4.5))
            ax = fig.add_axes((0.12, 0.15, 0.83, 0.78))

        if singular_values.size == 0 or np.max(singular_values) <= 0:
            ax.set_title("Singular values unavailable")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value")
            return suggestion

        normalized = singular_values / singular_values[0]
        indices = np.arange(1, singular_values.size + 1)
        ax.semilogy(
            indices, normalized, marker="o", linewidth=1.5, label="Normalized singular values"
        )

        for cond in conditioning_lines:
            if cond is None or cond <= 0:
                continue
            ax.axhline(
                1.0 / cond, linestyle="--", linewidth=0.8, alpha=0.5, label=f"cond={cond:.0e}"
            )

        if fit is not None:
            ax.semilogy(
                indices,
                fit["fit_curve"],
                color="tab:green",
                linestyle="-.",
                linewidth=1.2,
                label="Leading log-fit",
            )
            ax.axvline(
                fit["suggested_index"] + 1,
                color="tab:red",
                linestyle=":",
                linewidth=1.2,
                label=f"turnoff idx={fit['suggested_index'] + 1}",
            )

        if suggestion is not None and suggestion > 0:
            ax.axhline(
                1.0 / suggestion,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"suggested={suggestion:.2e}",
            )

        ax.set_title("Normalized IM singular values")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value / max singular value")
        ax.legend(loc="best", fontsize="small")
        return suggestion

    def _compute_inverse_from_svd(self, matrix, method: str, conditioning, tikhonov_reg: float):
        matrix = np.asarray(matrix, dtype=np.float64)
        num_modes = matrix.shape[1]

        if matrix.size == 0:
            return (
                np.zeros((num_modes, matrix.shape[0]), dtype=self.cm.dtype),
                np.array([], dtype=np.float64),
                np.array([], dtype=bool),
            )

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] <= 0:
            return (
                np.zeros((num_modes, matrix.shape[0]), dtype=self.cm.dtype),
                singular_values,
                np.zeros_like(singular_values, dtype=bool),
            )

        U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
        retained = singular_values > 0
        if conditioning is not None:
            retained &= singular_values >= (singular_values[0] / conditioning)

        inverse_singular_values = np.zeros_like(singular_values)
        if method == "svd":
            inverse_singular_values[retained] = 1.0 / singular_values[retained]
        else:
            if tikhonov_reg < 0:
                raise ValueError("tikhonov_reg must be non-negative")
            inverse_singular_values[retained] = singular_values[retained] / (
                singular_values[retained] ** 2 + tikhonov_reg**2
            )

        inverse = (Vh.T * inverse_singular_values) @ U.T
        return inverse.astype(self.cm.dtype, copy=False), singular_values, retained

    def compute_cm(self, method=None, num_dropped_modes=None, conditioning=None, tikhonov_reg=None):
        """
        Compute the control matrix from the interaction matrix.

        Parameters
        ----------
        method : str, optional
            Inversion method to use. Supported values are ``svd`` and
            ``tikhonov``. Defaults to the configured ``cm_method``.
        num_dropped_modes : int, optional
            Number of modal commands to suppress before inversion. Defaults to
            the configured ``num_dropped_modes``.
        conditioning : float, optional
            Optional target conditioning number. Singular values below
            ``max(s) / conditioning`` are discarded.
        tikhonov_reg : float, optional
            Tikhonov regularization strength used when ``method`` is
            ``tikhonov``. Defaults to the configured ``tikhonov_reg``.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            method = self._validate_cm_method(self.cm_method if method is None else method)
            requested_dropped_modes = (
                self.num_dropped_modes if num_dropped_modes is None else int(num_dropped_modes)
            )
            requested_conditioning = self.conditioning if conditioning is None else conditioning
            requested_tikhonov = self.tikhonov_reg if tikhonov_reg is None else float(tikhonov_reg)

            if requested_conditioning is not None:
                requested_conditioning = float(requested_conditioning)
                if requested_conditioning <= 1:
                    raise ValueError("conditioning must be greater than 1 when provided")

            self.num_dropped_modes = requested_dropped_modes
            self.cm_method = method
            self.conditioning = requested_conditioning
            self.tikhonov_reg = requested_tikhonov
            self.num_active_modes = self.num_modes - self.num_dropped_modes
            if self.num_active_modes < 0:
                raise ValueError("Invalid number of modes used in CM. Check num_dropped_modes")
            active_im = self.im[:, : self.num_active_modes]
            inverse, singular_values, retained = self._compute_inverse_from_svd(
                active_im,
                method=self.cm_method,
                conditioning=self.conditioning,
                tikhonov_reg=self.tikhonov_reg,
            )

            self.cm[:, :] = 0
            self.cm[: self.num_active_modes, :] = inverse
            self.cm[self.num_active_modes :, :] = 0
            self.g_cm = self.gain * self.cm
            self.f_im = np.copy(self.im)
            self.f_im[:, self.num_active_modes :] = 0
            self.last_singular_values = singular_values
            self.last_retained_singular_mask = retained
            suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
            self.last_suggested_conditioning = suggestion
            self.last_singular_value_fit = fit
            component_logger.info(
                "Computed control matrix method=%s active_modes=%s dropped_modes=%s conditioning=%s retained_singular_values=%s tikhonov_reg=%s",
                self.cm_method,
                self.num_active_modes,
                self.num_dropped_modes,
                self.conditioning,
                int(np.count_nonzero(retained)),
                self.tikhonov_reg,
            )
        except Exception:
            component_logger.exception("Failed to compute control matrix")
            raise
        return

    # @jit(nopython=True)
    def update_correction_pol(
        self, correction=np.array([], dtype=np.float32), slopes=np.array([], dtype=np.float32)
    ):
        """
        Update the correction using pseudo open loop slopes.

        Parameters
        ----------
        correction : numpy.ndarray
            Current correction vector.
        slopes : numpy.ndarray
            Current slopes vector.

        Returns
        -------
        numpy.ndarray
            Updated correction vector.
        """
        # Compute POL Slopes s_{POL} = s_{RES} + im*c_{n-1}
        # print(f'slopes: {slopes.shape}, im: {self.im.shape}, corr: {correction.shape}')
        s_pol = slopes - self.f_im @ correction

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1 - self.gain) * correction - np.dot(self.g_cm, s_pol)

    def standard_integrator_pol(self):
        """
        Standard integrator using the pseudo open loop slopes.
        """
        residual_slopes = self.read_stream("signal", out=self._signal_buffer)
        current_correction = self.read_stream("wfc", block=False, out=self._wfc_buffer)
        # print(f'slopes: {residual_slopes.shape}, im: {self.im.shape}, corr: {current_correction.shape}')

        new_correction = self.update_correction_pol(
            correction=current_correction, slopes=residual_slopes
        )
        new_correction[self.num_active_modes :] = 0
        self.send_to_wfc(new_correction)

        return

    def standard_integrator(self):
        """
        Standard integrator.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        new_correction = leaky_integrator_numba(
            slopes,
            self.g_cm,
            self.read_stream("wfc", block=False, out=self._wfc_buffer).squeeze(),
            self._correction_buffer,
            np.float32(0),  # No leak
            self.num_active_modes,
        )
        self.send_to_wfc(new_correction, slopes=slopes)
        return

    def leaky_integrator(self):
        """
        Leaky integrator.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        new_correction = leaky_integrator_numba(
            slopes,
            self.g_cm,
            self.read_stream("wfc", block=False, out=self._wfc_buffer).squeeze(),
            self._correction_buffer,
            np.float32(self.leaky_gain),
            self.num_active_modes,
        )
        self.send_to_wfc(new_correction, slopes=slopes)
        return

    def pid_integrator_pol(self):
        """
        PID integrator using the pseudo-open loop slopes.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        correction = self.read_stream("wfc", block=False, out=self._wfc_buffer)
        pol_slopes = slopes - self.f_im @ correction
        return self.pid_integrator(slopes=pol_slopes, correction=correction)

    def pid_integrator(self, slopes=None, correction=None):
        """
        PID integrator.

        Parameters
        ----------
        slopes : numpy.ndarray, optional
            Current slopes vector. If not provided, reads from shared memory.
        correction : numpy.ndarray, optional
            Current correction vector. If not provided, reads from shared memory.
        """
        if slopes is None:
            slopes = self.read_stream("signal")
        if correction is None:
            correction = self.read_stream("wfc", block=False)

        # Compute raw error term (numba accelerated)
        wf_error = comp_correction(cm=self.cm, slopes=slopes)
        derivative = wf_error - self.previous_wf_error

        # Apply low-pass filter to the derivative to reduce noise
        derivative = (
            self.derivative_filter * derivative
            + (1 - self.derivative_filter) * self.previous_derivative
        )

        # Update integral (anti-windup: conditional integration)
        # not_output_limiting = self.control_limits[0] is None or self.control_limits[1] is None
        is_clipped = np.any(self.control_output == self.control_limits[0]) or np.any(
            self.control_output == self.control_limits[1]
        )
        # Check to make sure we aren't actively clipping the correction
        if not is_clipped:
            # Add to integral
            self.integral += wf_error
            # Clip integral term
            self.integral = np.clip(self.integral, *self.integral_limits)

        # Calculate PID output
        control_output = (
            self.p_gain * wf_error + self.i_gain * self.integral + self.d_gain * derivative
        )

        control_output = np.clip(control_output, *self.control_limits)

        # Get new correction vector from the control output
        new_correction = (
            1 - self.leaky_gain
        ) * correction - control_output  # Negative control direction is convention for pyrtc

        # Remove anything in non-corrected modes (might be redundant)
        new_correction[self.num_active_modes :] = 0

        # Clip correction (force the loop to not over correct a mode)
        new_correction = np.clip(new_correction, *self.absolute_limits)

        # Apply new correction to mirror
        self.send_to_wfc(new_correction, slopes=slopes)

        # Save state for next iteration
        self.previous_wf_error = wf_error
        self.previous_derivative = derivative
        self.control_output = control_output

        return

    def send_to_wfc(self, correction, slopes=None):
        # Get an initial slope reading to set shapes
        correction = correction.reshape(self.flat.shape)
        if self.cl_docrime and isinstance(slopes, np.ndarray):
            slopes = slopes.reshape(slopes.size, 1)
            # Compute new random shape
            rand_shape = (
                np.random.uniform(-self.poke_amp, self.poke_amp, correction.size)
                .astype(self.docrime_buffer[0].dtype)
                .reshape(self.docrime_buffer[0].shape)
            )

            # Adds to end of buffer (i.e. pos -1)
            add_to_buffer(self.docrime_buffer, rand_shape)

            rand_shape = rand_shape.astype(correction.dtype).reshape(correction.shape)

            # Only add randomness to active modes, otherwise it will build up
            if self.num_active_modes > 0:
                correction[: self.num_active_modes] += rand_shape[: self.num_active_modes]
                correction[self.num_active_modes :] = rand_shape[self.num_active_modes :]
            else:
                correction = rand_shape

            # Send our new pertubation to the WFC
            self.write_stream("wfc", correction)

            # Correlate Current response with old correction by delay time
            self.docrime_cross += slopes @ self.docrime_buffer[0].T
            self.docrime_auto += self.docrime_buffer[0] @ self.docrime_buffer[0].T

            self.num_iters_dc += 1

        else:
            self.write_stream("wfc", correction)
        return

    def solve_docrime(self):

        component_logger = getattr(self, "logger", logger)
        try:
            self.cl_dcim = (self.docrime_cross / self.num_iters_dc) @ np.linalg.inv(
                self.docrime_auto / self.num_iters_dc
            )
            tmp_file_path = get_tmp_filepath(self.im_file, unique_str="CL_docrime")
            component_logger.info("Saving DOCRIME matrix to %s", tmp_file_path)
            np.save(tmp_file_path, self.cl_dcim)
        except Exception:
            component_logger.exception("Failed to solve DOCRIME interaction matrix")
            raise

        return

    def plot_im(self):
        """Plot the interaction matrix. Returns the figure (not shown)."""
        fig, ax = pyplot().subplots()
        im = ax.imshow(self.im, cmap="inferno", aspect="auto")
        fig.colorbar(im, ax=ax)
        return fig


if __name__ == "__main__":
    launch_component(Loop, "loop", start=False)
