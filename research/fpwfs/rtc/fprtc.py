"""Simulated focal-plane camera + DM for pyRTC (research components, ``class_file``).

Modelled on ``pyrtc/hardware/hcipy_interface.py``: one shared simulation context
(:class:`FPSimContext`, a manager ``resources:`` entry) couples a wavefront-sensor
component (:class:`FPCamera`, ``wfs`` section) and a corrector component
(:class:`FPDM`, ``wfc`` section) in the soft-RTC process.

Physics (all from ``research/fpwfs/fpsim``, the offline harness):

* atmosphere: pyturb ``keck`` profile at 0.6", advanced by ``1 / frame_rate`` of
  simulated time per exposure (``fpsim.atmos.Turbulence``). ``atmosphere: live``
  steps pyturb inside every exposure; ``atmosphere: precomputed`` steps the same
  pyturb atmosphere (same seed, same dt) ahead of time into a GPU buffer, so the
  camera thread costs less and the run sees exactly the screens the offline
  harness sees.
* DM: the Keck 349-actuator Gaussian-influence DM (``fpsim.system.build``). The
  ``FPDM`` component turns the loop's 120 modal coefficients (metres, unit-rms
  DM-KL modes) into actuator commands with its M2C (the first 120 columns of the
  300-mode Keck basis), and the context renders OPD = IF @ actuators.
  Sign: the DM *adds* its OPD, residual = turbulence + DM, so a command ``c``
  on a flat wavefront reads back as ``+c`` (pyRTC's identity-IM convention).
* sensor: ``fpsim.sensor.FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0,
  photons=1e5))``: H-band, 64 x 64 px at Nyquist, 1 rad rms defocus, Poisson +
  0.6 e- read noise. Electrons are rounded to integer ADU (1 e-/ADU) on a
  ``bias`` offset (the WFS dark is set to the bias, so the ``wfs`` stream is the
  rounded electron image, negative read-noise values kept).
* truth: per exposure, the H-band short-exposure Strehl
  (``fpsim.loop.h_band_science``, peak ratio), the pupil rms of the residual, the
  ideal 300-mode projection of the residual (``ModalProjector``), and an
  accumulated long-exposure PSF after ``le_start`` frames. Everything after the
  atmosphere is one captured CUDA graph.

Timing model: an exposure's frame is published one frame period after the DM
state it saw was sampled (the camera reads out at the end of the exposure): at
each tick the camera publishes frame e - 1, then samples the DM for exposure e
and renders it. With a publish-to-DM latency below one period, the command
computed from frame k first shapes exposure k + 2: the offline harness's
``delay = 2``; above it, delay 3. The context logs, for
every exposure, which frame's command was on the DM, so the realised delay is
measured, not assumed.

Streams follow pyRTC's convention: frames are rendered (H, W) and stored
transposed, as (width, height).

Warm start / hand-over (the network only holds a loop that is already closed):
for the first ``warm_frames`` exposures the context runs the ideal sensor
itself: from each exposure's true residual it takes the projection on the first
120 modes and integrates ``c <- leak c - gain est`` (pyRTC sign, same gain and
leak as the loop), writing ``c`` to the ``wfc`` stream at the moment that
exposure's frame is published (so the warm loop has the same 2-frame delay).
The pyRTC ``Loop`` is paused meanwhile; the network runs in shadow (the
reconstructor publishes ``signal`` from the first frame). At exposure
``warm_frames`` the context stops writing ``wfc`` and calls ``on_handover``
(the run script's hook that starts the loop); the loop then integrates the
network's signal on top of the last warm command already in ``wfc``.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np

from pyrtc.streams import open_stream
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor

FPWFS = pathlib.Path(__file__).resolve().parents[1]
if str(FPWFS) not in sys.path:
    sys.path.insert(0, str(FPWFS))

DEFAULTS: dict[str, Any] = {
    "device": "cuda:0",  # simulation GPU (torch device; pyturb's CuPy uses the same index)
    "seed": 4000,  # Turbulence seed block (fpsim.atmos.Turbulence: atmospheres seed * 1000 + i)
    "atm_index": 0,  # which atmosphere of the block (i), as in a batched harness run
    "seeing": 0.6,
    "profile": "keck",
    "frame_rate": 1000.0,  # simulated frame rate: the atmosphere advances 1/frame_rate per exposure
    "wall_rate": 0.0,  # wall-clock frame rate of the camera thread (0: free-running)
    "atmosphere": "precomputed",  # or "live"
    "max_frames": 6000,  # exposures in the run (and in the precomputed buffer)
    "n_control": 120,
    "photons": 1.0e5,
    "defocus_rad": 1.0,
    "read_noise": 0.6,
    "bias": 100,
    "warm_frames": 300,
    "warm_gain": 0.4,
    "warm_leak": 0.99,
    "le_start": 600,  # long-exposure PSF accumulates from this exposure on (as exp13's settle)
    "cuda_graph": True,
    "strehl_every": 1,  # science Strehl / LE PSF on every N-th exposure (1: all; decimate to save GPU time)
    "wfc_stream": "wfc",  # the corrector stream the warm-start integrator writes
}


def _torch():
    import torch

    return torch


class FPSimContext:
    """Shared focal-plane simulation: atmosphere, DM, sensor, truth, logs."""

    def __init__(self, resource_conf: Mapping[str, Any] | None = None, system_conf=None):
        conf = dict(resource_conf or {})
        for key in ("class_name", "class_file"):
            conf.pop(key, None)
        unknown = set(conf) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown FPSimContext parameters: {sorted(unknown)}")
        self.p = {**DEFAULTS, **conf}
        if system_conf is not None and "wfc" in system_conf:
            # the warm start writes the corrector stream the loop drives
            self.p["wfc_stream"] = (system_conf["wfc"].get("output_streams") or {}).get(
                "wfc", self.p["wfc_stream"]
            )
        self.lock = threading.Lock()
        self.components: dict[str, Any] = {}
        self.on_handover = None  # set by the run script: starts the pyRTC loop
        self.handover_wall_time = None
        self.done = threading.Event()
        self.started = threading.Event()
        self._build()

    # -- construction -------------------------------------------------------------

    def _build(self) -> None:
        torch = _torch()
        from fpsim import system as S
        from fpsim.loop import DM, h_band_science
        from fpsim.sensor import FocalPlaneSensor, FPSensorConfig

        p = self.p
        self.dev = torch.device(p["device"])
        torch.cuda.set_device(self.dev)
        self.cfg = cfg = S.KeckConfig()
        sysd = S.build(cfg)
        self.n = cfg.n_pupil
        dev = self.dev
        self.pupil = torch.tensor(sysd["pupil"], device=dev)
        self.ifs = torch.tensor(sysd["ifs"], device=dev)  # (n*n, n_act)
        self.m2c = np.asarray(sysd["m2c"], dtype=np.float32)  # (n_act, 300)
        self.n_act = self.m2c.shape[0]
        self.n_modes = self.m2c.shape[1]
        self.n_control = int(p["n_control"])
        dm = DM(self.ifs, torch.tensor(self.m2c, device=dev), self.n)
        # ideal-sensor projection with a static index set (capturable)
        inside = (self.pupil.reshape(-1) > 0.5).nonzero().squeeze(1)
        self.inside = inside
        a = dm.surfaces[:, inside].T.double()
        self.pinv = torch.linalg.pinv(a).float()  # (n_modes, npts)
        self.sensor = FocalPlaneSensor(
            FPSensorConfig(
                defocus_rad=float(p["defocus_rad"]),
                photons=float(p["photons"]),
                read_noise=float(p["read_noise"]),
            ),
            self.pupil,
            cfg.grid_m,
        ).to(dev)
        self.npix = self.sensor.cfg.npix
        self.science = h_band_science(self.pupil, cfg.grid_m)
        with torch.no_grad():
            self.ref_peak = self.science(torch.zeros_like(self.pupil)).amax()
        self.dt = 1.0 / float(p["frame_rate"])
        self.max_frames = int(p["max_frames"])
        self.wall_period = 1.0 / float(p["wall_rate"]) if float(p["wall_rate"]) > 0 else 0.0
        self.bias = int(p["bias"])

        # atmosphere
        from fpsim.atmos import Turbulence

        with self._cupy_device():
            self.turb = Turbulence(cfg, batch=1, seed=int(p["seed"]), seeing=float(p["seeing"]),
                                   profile=str(p["profile"]))
            if int(p["atm_index"]):
                from fpsim.atmos import make_atmosphere

                self.turb.seeds = [int(p["seed"]) * Turbulence.SEED_STRIDE + int(p["atm_index"])]
                self.turb.atms = [make_atmosphere(cfg, seeing=float(p["seeing"]), profile=str(p["profile"]),
                                                  seed=self.turb.seeds[0])]
        self.precomputed = None
        if p["atmosphere"] == "precomputed":
            t0 = time.perf_counter()
            buf = torch.empty((self.max_frames, self.n, self.n), device=dev)
            with self._cupy_device():
                for k in range(self.max_frames):
                    buf[k] = self.turb.step(self.dt)[0]
            torch.cuda.synchronize(dev)
            self.precomputed = buf
            self.precompute_seconds = time.perf_counter() - t0
        elif p["atmosphere"] != "live":
            raise ValueError("atmosphere must be 'live' or 'precomputed'")

        # static graph inputs / outputs
        self.stream = torch.cuda.Stream(device=dev)
        self.screen_in = torch.zeros((self.n, self.n), device=dev)
        self.act_in = torch.zeros(self.n_act, device=dev)
        self.le_weight = torch.zeros((), device=dev)
        self.le_acc = torch.zeros((self.science.npix, self.science.npix), device=dev)
        self._act_host = torch.zeros(self.n_act, pin_memory=True)
        self._frame_host = torch.zeros((self.npix, self.npix), dtype=torch.int32, pin_memory=True)
        self._stats_host = torch.zeros(2 + self.n_modes, pin_memory=True)
        self.graph = None
        self.graph_active = False
        self.le_frames = 0
        self._capture()

        # host-side state
        self.dm_act = np.zeros(self.n_act, dtype=np.float32)  # latest actuator command (metres)
        self.dm_fid = 0  # frame id of the command on the DM (0: none yet)
        self.warm_cmd = np.zeros(self.n_control, dtype=np.float32)
        self.exposure = 0  # exposures rendered so far (frame ids are 1-based)
        self.pending = None  # (raw uint16 (W, H), frame id, ideal est (n_modes,)) awaiting readout
        self.next_tick = None
        self.missed_ticks = 0
        self.phase = "hold"  # hold -> warm -> network
        self.wfc_writer = None
        N = self.max_frames + 2
        self.log = {
            "t_tick": np.full(N, np.nan),
            "t_snap": np.full(N, np.nan),  # DM state sampled for exposure e
            "t_written": np.full(N, np.nan),  # frame e written to wfs_raw and wfs
            "t_publish": np.full(N, np.nan),  # frame e published (at tick e + 1)
            "t_render": np.full(N, np.nan),  # camera-sim time for exposure e (s)
            "strehl": np.full(N, np.nan),
            "rms_nm": np.full(N, np.nan),
            "dm_fid": np.zeros(N, dtype=np.int64),  # frame whose command shaped exposure e
            "network": np.zeros(N, dtype=bool),
            "t_dm": np.full(N, np.nan),  # when the command computed from frame e reached the DM
            "ideal_nm": np.zeros((N, self.n_control), dtype=np.float32),  # true residual, modes
        }

    def _cupy_device(self):
        import contextlib

        try:
            import cupy
        except ImportError:
            return contextlib.nullcontext()
        return cupy.cuda.Device(self.dev.index or 0)

    def _render(self, science: bool = True):
        """Graph body: residual -> (frame int32 (W, H), [strehl, rms, ideal modes])."""
        torch = _torch()
        residual = self.screen_in + (self.ifs @ self.act_in).reshape(self.n, self.n)
        e = self.sensor.frame(residual[None])[0]
        raw = (e.round() + self.bias).clamp(0, 65535).to(torch.int32).T.contiguous()
        flat = residual.reshape(-1).index_select(0, self.inside)
        flat = flat - flat.mean()
        rms = flat.std()
        est = self.pinv @ flat
        if science:
            psf = self.science(residual)
            self.le_acc.add_(psf * self.le_weight)
            strehl = psf.amax() / self.ref_peak
        else:
            strehl = torch.full((), float("nan"), device=residual.device)
        return raw, torch.cat([strehl[None], rms[None] * 1e9, est])

    def _capture(self) -> None:
        torch = _torch()
        with torch.cuda.stream(self.stream), torch.no_grad():
            for _ in range(3):
                self._render()
        self.stream.synchronize()
        self.le_acc.zero_()
        if not self.p["cuda_graph"]:
            return
        try:
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, stream=self.stream, capture_error_mode="thread_local"):
                self._g_raw, self._g_stats = self._render()
            self.graph = g
            self.graph_lite = None
            if int(self.p["strehl_every"]) > 1:
                gl = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(gl, stream=self.stream, capture_error_mode="thread_local"):
                    self._gl_raw, self._gl_stats = self._render(science=False)
                self.graph_lite = gl
            self.graph_active = True
        except Exception as exc:  # pragma: no cover - fallback
            print(f"FPSimContext: CUDA graph capture failed ({exc}); eager", flush=True)
            self.graph = None
        self.le_acc.zero_()

    # -- registration -------------------------------------------------------------

    def register_component(self, role: str, component) -> None:
        self.components[role] = component

    # -- DM -------------------------------------------------------------------------

    def set_actuators(self, shape, frame_id: int) -> None:
        """Called by FPDM.send_to_hardware (corrector thread)."""
        now = time.perf_counter()
        with self.lock:
            np.copyto(self.dm_act, shape, casting="unsafe")
            self.dm_fid = int(frame_id or 0)
        fid = int(frame_id or 0)
        if 0 < fid < len(self.log["t_dm"]) and np.isnan(self.log["t_dm"][fid]):
            self.log["t_dm"][fid] = now

    # -- camera ---------------------------------------------------------------------

    def begin(self) -> None:
        """Leave the hold state: the next exposure is exposure 1 of the warm phase."""
        self.next_tick = time.perf_counter()
        self.phase = "warm" if int(self.p["warm_frames"]) > 0 else "network"
        self.started.set()

    def _wait_tick(self) -> float:
        if not self.wall_period:
            return time.perf_counter()
        target = self.next_tick
        now = time.perf_counter()
        if target - now > 3e-4:
            time.sleep(target - now - 3e-4)
        while time.perf_counter() < target:
            time.sleep(0)  # releases the GIL while spinning
        now = time.perf_counter()
        self.next_tick = target + self.wall_period
        if now > self.next_tick:  # overran a whole period: do not burst to catch up
            self.missed_ticks += int((now - target) / self.wall_period)
            self.next_tick = now + self.wall_period
        return now

    def camera_tick(self):
        """Start a frame period: sample the DM state for the new exposure.

        Returns ``(True, pending)`` where ``pending`` is the previous exposure's
        frame to read out now (``None`` before the first one), or ``(False, None)``
        when the camera is holding or the run is over.
        """
        if self.phase == "hold" or self.done.is_set():
            time.sleep(1e-3)
            return False, None
        t = self._wait_tick()
        e = self.exposure + 1
        if e > self.max_frames:
            self.done.set()
            return False, None
        self.log["t_tick"][e] = t
        self._tick_exposure = e
        return True, self.pending

    def after_publish(self, published) -> None:
        """Warm-phase control from the frame just read out, then render the new exposure."""
        torch = _torch()
        if published is not None:
            _, fid, est = published
            self.log["t_publish"][fid] = time.perf_counter()
            if self.phase == "warm":
                p = self.p
                self.warm_cmd *= np.float32(p["warm_leak"])
                self.warm_cmd -= np.float32(p["warm_gain"]) * est[: self.n_control]
                if self.wfc_writer is None:
                    self.wfc_writer = open_stream(self.p["wfc_stream"])
                self.wfc_writer.write(self.warm_cmd, frame_id=fid)
                if fid >= int(p["warm_frames"]):
                    self.phase = "network"
                    self.handover_frame = fid
                    self.handover_wall_time = time.perf_counter()
                    if self.on_handover is not None:
                        threading.Thread(target=self.on_handover, daemon=True).start()
        e = self._tick_exposure
        # Sample the DM for exposure e right after frame e - 1 is out: a command
        # computed from frame e - 2 shapes exposure e if it landed within one frame
        # period of that frame's publication (the harness's 2-frame delay).
        with self.lock:
            np.copyto(self._act_host.numpy(), self.dm_act)
            self.log["dm_fid"][e] = self.dm_fid
        self.log["t_snap"][e] = time.perf_counter()
        self.log["network"][e] = self.phase == "network"
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream), torch.no_grad():
            if self.precomputed is not None:
                self.screen_in.copy_(self.precomputed[e - 1])
            else:
                with self._cupy_device():
                    self.screen_in.copy_(self.turb.step(self.dt)[0])
            self.act_in.copy_(self._act_host, non_blocking=True)
            self.le_weight.fill_(1.0 if e > int(self.p["le_start"]) else 0.0)
            science = (e % int(self.p["strehl_every"])) == 0
            if science and e > int(self.p["le_start"]):
                self.le_frames += 1
            if self.graph is not None and (science or self.graph_lite is None):
                self.graph.replay()
                raw, stats = self._g_raw, self._g_stats
            elif self.graph is not None:
                self.graph_lite.replay()
                raw, stats = self._gl_raw, self._gl_stats
            else:
                raw, stats = self._render(science)
            self._frame_host.copy_(raw, non_blocking=True)
            self._stats_host.copy_(stats, non_blocking=True)
        self.stream.synchronize()
        self.log["t_render"][e] = time.perf_counter() - t0
        st = self._stats_host.numpy()
        self.log["strehl"][e] = st[0]
        self.log["rms_nm"][e] = st[1]
        est = st[2:].copy()
        self.log["ideal_nm"][e] = est[: self.n_control] * 1e9
        self.pending = (self._frame_host.numpy().astype(np.uint16), e, est)
        self.exposure = e

    # -- results --------------------------------------------------------------------

    def long_exposure_strehl(self) -> float:
        n = self.le_frames
        if n == 0:
            return float("nan")
        torch = _torch()
        torch.cuda.synchronize(self.dev)
        return float(self.le_acc.amax() / n / self.ref_peak)

    def save_log(self, path) -> None:
        n = self.exposure + 1
        out = {k: v[:n] for k, v in self.log.items()}
        out["le_strehl"] = np.array(self.long_exposure_strehl())
        out["missed_ticks"] = np.array(self.missed_ticks)
        out["handover_frame"] = np.array(getattr(self, "handover_frame", -1))
        np.savez_compressed(path, **out)


def _unwrap(resource):
    return getattr(resource, "context", resource)


class FPCamera(WavefrontSensor):
    """``wfs`` section: publishes the simulated focal-plane frames."""

    EXTRA_CONFIG_KEYS = ()

    def __init__(self, conf, context) -> None:
        self.context = _unwrap(context)
        conf = dict(conf)
        conf["width"] = conf["height"] = self.context.npix
        super().__init__(conf)
        self.context.register_component("wfs", self)
        self.set_dark(np.full(self.image_raw_shape, self.context.bias, dtype=np.int32))
        self._wfs_buffer = np.empty(self.image_shape, dtype=self.image_dtype)

    def expose(self):
        ctx = self.context
        active, published = ctx.camera_tick()
        if not active:
            return
        if published is not None:
            raw, fid, _ = published
            self.frame_id = fid
            self.write_stream("wfs_raw", raw)
            np.subtract(raw, self.dark, out=self._wfs_buffer, casting="unsafe")
            self.write_stream("wfs", self._wfs_buffer)
            ctx.log["t_written"][fid] = time.perf_counter()
        ctx.after_publish(published)


class FPDM(WavefrontCorrector):
    """``wfc`` section: 120 modal commands -> 349 actuators -> the simulated DM."""

    EXTRA_CONFIG_KEYS = ()

    def __init__(self, conf, context) -> None:
        self.context = _unwrap(context)
        conf = dict(conf)
        conf["num_actuators"] = self.context.n_act
        conf["num_modes"] = self.context.n_control
        super().__init__(conf)
        self.set_m2c(self.context.m2c[:, : self.context.n_control].copy())
        self.m2c_source = "fpsim-keck-dmkl"
        self.context.register_component("wfc", self)

    def send_to_hardware(self):
        super().send_to_hardware()
        self.context.set_actuators(self.current_shape, self.frame_id)
