"""Drive several wavefront correctors from one loop (woofer/tweeter, tip-tilt offload).

The loop controls one modal vector. A :class:`CorrectorSplitter` in the
loop's ``wfc`` section owns that stream. It is the concatenation of each
corrector's modes, in the order they are listed, and the splitter writes
each corrector its slice every frame. The interaction matrix therefore
calibrates every corrector's modes, because each poke reaches the right
device, and the loop needs no changes.

Offloading moves content from one corrector (the *source*, e.g. a tweeter)
to another (the *target*, e.g. a woofer or a tip-tilt stage) without changing
the wavefront. The coupling matrix ``C`` (source modes x target modes) gives
the source-mode equivalent of one unit of each target mode.
:meth:`CorrectorSplitter.set_coupling_from_im` estimates it from the loop's
IM as ``pinv(IM_source) @ IM_target``. Each frame, with offload gain ``g``:

    source_out = source_loop - C @ offload
    target_out = target_loop + offload
    offload   += g * pinv(C) @ source_out

The target takes over, at rate ``g``, the part of the source command it can
represent, and the source is left with the rest. A configured ``gain``
takes effect once the coupling is set, so calibrate the IM first (its pokes
must reach the correctors unchanged), then call ``set_coupling_from_im``.

Config (``wfc`` section)::

    wfc:
      class_name: pyrtc.corrector_splitter.CorrectorSplitter
      correctors:
        - {name: woofer, stream: woofer_wfc, modes: 20}
        - {name: tweeter, stream: tweeter_wfc, modes: 60}
      offload: {source: tweeter, target: woofer, gain: 0.02}   # optional
      functions: [split]

Each corrector is an ordinary wavefront-corrector section whose ``wfc``
input stream is the listed ``stream``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pyrtc.component import Component
from pyrtc.logging_utils import get_logger
from pyrtc.streams import create_stream, open_stream
from pyrtc.utils import set_from_config

logger = get_logger(__name__)


class CorrectorSplitter(Component):
    """Split the loop's modal command across several correctors, with offloading."""

    EXTRA_CONFIG_KEYS = ("correctors", "offload", "num_modes")

    def __init__(self, conf) -> None:
        self.targets = self._parse_correctors(conf.get("correctors"))
        self.num_modes = sum(target["modes"] for target in self.targets)
        configured = conf.get("num_modes")
        if configured is not None and int(configured) != self.num_modes:
            raise ValueError(
                f"wfc.num_modes ({configured}) must equal the correctors' total ({self.num_modes})"
            )
        super().__init__(conf)
        self.correction_vector = create_stream(
            self.output_stream_name("wfc"), (self.num_modes,), np.float32
        )
        self.register_output_stream("wfc", self.correction_vector)
        self.register_input_stream("wfc", self.correction_vector)
        self._buffer = np.empty(self.num_modes, dtype=np.float32)
        self._outputs: dict[str, Any] = {}
        self._missing_logged: set[str] = set()

        self.offload_source = None
        self.offload_target = None
        self.offload_gain = 0.0
        self.coupling = None
        self._coupling_pinv = None
        self.offload_state = None
        offload = set_from_config(conf, "offload", None)
        if offload:
            self._configure_offload(offload)
        self.logger.info(
            "Initialized corrector splitter: %s",
            ", ".join(f"{t['name']}={t['modes']} modes -> {t['stream']}" for t in self.targets),
        )

    # -- configuration --------------------------------------------------------

    @staticmethod
    def _parse_correctors(raw) -> list[dict[str, Any]]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("wfc.correctors must be a non-empty list of {name, stream, modes}")
        targets = []
        start = 0
        names = set()
        for index, entry in enumerate(raw):
            if not isinstance(entry, Mapping) or "stream" not in entry or "modes" not in entry:
                raise ValueError(f"wfc.correctors[{index}] needs 'stream' and 'modes'")
            modes = int(entry["modes"])
            if modes < 1:
                raise ValueError(f"wfc.correctors[{index}].modes must be >= 1")
            name = str(entry.get("name", entry["stream"]))
            if name in names:
                raise ValueError(f"duplicate corrector name {name!r}")
            names.add(name)
            targets.append(
                {"name": name, "stream": str(entry["stream"]), "modes": modes, "start": start}
            )
            start += modes
        return targets

    def _target(self, name) -> dict[str, Any]:
        for target in self.targets:
            if target["name"] == str(name):
                return target
        raise ValueError(f"unknown corrector {name!r}; have {[t['name'] for t in self.targets]}")

    def _configure_offload(self, offload: Mapping[str, Any]) -> None:
        source = self._target(offload.get("source"))
        target = self._target(offload.get("target"))
        if source is target:
            raise ValueError("offload source and target must differ")
        self.offload_source, self.offload_target = source, target
        coupling = offload.get("coupling")
        if isinstance(coupling, str) and coupling:
            coupling = np.load(coupling)
        gain = float(offload.get("gain", 0.0))
        if coupling is not None:
            self.set_coupling(coupling)
            self.set_offload_gain(gain)
        else:
            # Applied once the coupling is known (e.g. set_coupling_from_im).
            self._pending_gain = gain

    def set_coupling(self, coupling) -> None:
        """Set ``C``: the source-mode equivalent of one unit of each target mode."""

        if self.offload_source is None:
            raise RuntimeError("configure 'offload' (source and target) first")
        coupling = np.asarray(coupling, dtype=np.float64)
        expected = (self.offload_source["modes"], self.offload_target["modes"])
        if coupling.shape != expected:
            raise ValueError(f"coupling must have shape {expected}, got {coupling.shape}")
        self.coupling = coupling
        self._coupling_pinv = np.linalg.pinv(coupling)
        self.offload_state = np.zeros(self.offload_target["modes"])
        pending = getattr(self, "_pending_gain", None)
        if pending is not None:
            self._pending_gain = None
            self.set_offload_gain(pending)

    def set_coupling_from_im(self, im) -> np.ndarray:
        """Estimate ``C`` from the loop IM (``signal_size x num_modes``) and set it.

        ``C = pinv(IM_source) @ IM_target``: the source-mode command that
        produces the same WFS signal as each target mode.
        """

        im = np.asarray(im, dtype=np.float64)
        if im.ndim != 2 or im.shape[1] != self.num_modes:
            raise ValueError(f"IM must have {self.num_modes} columns, got shape {im.shape}")
        source, target = self.offload_source, self.offload_target
        if source is None:
            raise RuntimeError("configure 'offload' (source and target) first")
        im_source = im[:, source["start"] : source["start"] + source["modes"]]
        im_target = im[:, target["start"] : target["start"] + target["modes"]]
        coupling = np.linalg.pinv(im_source) @ im_target
        self.set_coupling(coupling)
        return coupling

    def set_offload_gain(self, gain: float) -> None:
        """Set the offload gain (0 disables offloading; the state is kept)."""

        gain = float(gain)
        if not 0.0 <= gain <= 1.0:
            raise ValueError("offload gain must be in [0, 1]")
        if gain > 0 and self.coupling is None:
            raise RuntimeError("set the coupling (set_coupling or set_coupling_from_im) first")
        self.offload_gain = gain

    def reset_offload(self) -> None:
        """Hand everything back to the source (clear the offloaded state)."""

        if self.offload_state is not None:
            self.offload_state[:] = 0.0

    # -- runtime ----------------------------------------------------------------

    def _output(self, target) -> Any:
        stream = self._outputs.get(target["name"])
        if stream is None:
            try:
                stream = open_stream(target["stream"])
            except FileNotFoundError:
                if target["name"] not in self._missing_logged:
                    self._missing_logged.add(target["name"])
                    self.logger.warning(
                        "corrector stream %r does not exist yet; its commands are dropped",
                        target["stream"],
                    )
                return None
            self._outputs[target["name"]] = stream
        return stream

    def split_command(self, command) -> dict[str, np.ndarray]:
        """Return each corrector's command for one loop command (applies offload)."""

        command = np.asarray(command, dtype=np.float64).reshape(self.num_modes)
        parts = {
            target["name"]: command[target["start"] : target["start"] + target["modes"]].copy()
            for target in self.targets
        }
        if self.coupling is not None and self.offload_state is not None:
            source, target = self.offload_source["name"], self.offload_target["name"]
            parts[source] -= self.coupling @ self.offload_state
            parts[target] += self.offload_state
            if self.offload_gain > 0:
                self.offload_state += self.offload_gain * (self._coupling_pinv @ parts[source])
        return parts

    def split(self):
        """Worker function: forward the latest loop command to every corrector."""

        command = self.read_stream("wfc", out=self._buffer)
        parts = self.split_command(command)
        for target in self.targets:
            stream = self._output(target)
            if stream is None:
                continue
            stream.write(parts[target["name"]].astype(np.float32), frame_id=self.frame_id)

    def flatten(self):
        """Zero the loop command and the offloaded state."""

        self.reset_offload()
        self.write_stream("wfc", np.zeros(self.num_modes, dtype=np.float32))

    def close(self, **kwargs):
        for stream in self._outputs.values():
            try:
                stream.close()
            except Exception:
                pass
        self._outputs.clear()
        super().close(**kwargs)
