"""Wavefront-corrector abstractions and modal-to-zonal mapping helpers.

This module defines the base class used by pyrtc deformable mirrors and other
corrective devices. It manages command streams, flat handling, actuator masks,
and optional 2D layout views, while leaving hardware transport details to the
concrete adapter subclasses.
"""

import numpy as np
from numba import jit

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import create_stream
from pyrtc.component import Component
from pyrtc.modal_basis import BasisConfig, build_m2c, default_actuator_layout, parse_basis_config
from pyrtc.utils import gaussian_2d_grid, pyplot, set_from_config

logger = get_logger(__name__)


@jit(nopython=True)
def ModaltoZonalWithFlat(
    correction=np.array([], dtype=np.float32),
    M2C=np.array([[]], dtype=np.float32),
    flat=np.array([], dtype=np.float32),
):
    """Project a modal correction into actuator space and add the flat shape."""

    return M2C @ correction + flat


class WavefrontCorrector(Component):
    """
    Base class for deformable mirrors and other wavefront-correction devices.

    ``WavefrontCorrector`` is responsible for the control-plane machinery around
    command generation: SHM output, flat shapes, mode-to-command transforms,
    floating actuator handling, and delayed command buffers. Subclasses are left
    to implement the device-specific transport in ``send_to_hardware``.

    Config
    ------
    name : str
        Name of the wavefront corrector.
    num_actuators : int
        Number of actuators. Required.
    num_modes : int
        Number of modes. Required.
    affinity : str
        Affinity setting.
    m2c_file : str
        Path to the mode-to-command file (``.npy`` or ``.dat``). Takes
        precedence over ``basis``.
    basis : dict, optional
        Build ``M2C`` with aobasis from the actuator geometry, e.g.
        ``{type: kl, pupil_diameter: 8.0, r0: 0.16, L0: 30.0}``. Types are
        ``kl``, ``zernike``, ``fourier``, ``zonal``, ``zonal_fast`` and
        ``hadamard``; see :mod:`pyrtc.modal_basis`. Ignored when
        ``m2c_file`` is set. Without either, ``M2C`` is the identity.
    floating_influence_radius : int, optional
        Radius for floating influence. Default is 1.
    frame_delay : int, optional
        Frame delay. Default is 0.
    save_file : str, optional
        File to save the shape. Default is "wfc_shape.npy".

    Attributes
    ----------
    name : str
        Name of the wavefront corrector.
    num_actuators : int
        Number of actuators.
    num_modes : int
        Number of modes.
    affinity : str
        Affinity setting.
    m2c_file : str
        Path to the mode-to-command file.
    correction_vector : pyshmem.SharedMemory
        Correction vector.
    correction_vector_2d : pyshmem.SharedMemory or None
        2D correction vector for display.
    flat : numpy.ndarray
        Initial flat shape.
    flat_modal : numpy.ndarray
        Flat shape in modal basis.
    current_shape : numpy.ndarray
        Current shape.
    actuator_status : numpy.ndarray
        Status of each actuator.
    index_map : numpy.ndarray or None
        Index map for actuators.
    floating_influence_radius : int
        Radius for floating influence.
    float_matrix : numpy.ndarray
        Floating actuator matrix.
    frame_delay : int
        Frame delay.
    command_cap : float or None
        Optional absolute limit applied to actuator-space commands before they
        are handed to the hardware adapter.
    save_file : str
        File to save the shape.
    layout : numpy.ndarray or None
        Layout of the actuators.
    M2C : numpy.ndarray
        Mode-to-command matrix.
    f_M2C : numpy.ndarray
        Floating mode-to-command matrix.
    C2M : numpy.ndarray
        Command-to-mode matrix.
    current_correction : numpy.ndarray
        Current correction vector.
    shape_buffer : numpy.ndarray
        Buffer for shapes with frame delay.
    correction_vector_2d_template : numpy.ndarray
        Template for the 2D correction vector.
    """

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.name = conf["name"]
            self.num_actuators = conf["num_actuators"]
            self.num_modes = conf["num_modes"]
            self.m2c_file = set_from_config(conf, "m2c_file", "")
            self.basis_conf = parse_basis_config(conf.get("basis"), num_modes=self.num_modes)
            self.m2c_source = "identity"
            self._basis_layout = None

            self.correction_vector = create_stream(
                self.output_stream_name("wfc"),
                (self.num_modes,),
                np.float32,
                gpu_device=self.gpu_device,
            )
            self.register_output_stream("wfc", self.correction_vector)
            # The corrector also consumes the commands the loop writes to the
            # same stream; as a registered input it forwards their frame id
            # to ``wfc_2d``.
            self.register_input_stream("wfc", self.correction_vector)
            # Pre-allocated hot-path read buffer (ignored for GPU streams).
            self._wfc_buffer = np.empty((self.num_modes,), dtype=np.float32)
            self.correction_vector_2d = None

            self.set_layout(None)

            self.flat = np.zeros(self.num_actuators, dtype=np.float32)
            self.flat_modal = np.zeros(self.num_modes, dtype=self.flat.dtype)
            self.current_shape = np.zeros_like(self.flat)
            self.flat_file = set_from_config(conf, "flat_file", "")
            self.command_cap = set_from_config(conf, "command_cap", None)
            self.load_flat()

            self.actuator_status = np.array([True] * self.num_actuators)
            self.index_map = None
            self.floating_influence_radius = set_from_config(conf, "floating_influence_radius", 1)
            self.float_matrix = np.eye(self.num_actuators, dtype=self.flat.dtype)

            self.set_delay(set_from_config(conf, "frame_delay", 0))

            self.save_file = set_from_config(conf, "save_file", "wfc_shape.npy")
            self.read_m2c()
            self.logger.info(
                "Initialized wavefront corrector name=%s actuators=%s modes=%s command_cap=%s",
                self.name,
                self.num_actuators,
                self.num_modes,
                self.command_cap,
            )
        except Exception:
            logger.exception("Failed to initialize wavefront corrector")
            raise

        return

    def set_flat(self, flat):
        """
        Set the flat shape.

        Parameters
        ----------
        flat : numpy.ndarray
            Flat shape to set.
        """
        try:
            self.flat = flat.astype(self.flat.dtype)
            self.logger.info("Updated flat shape")
        except Exception:
            self.logger.exception("Failed to update flat shape")
            raise
        return

    def load_flat(self, filename=""):
        """
        Loads the Flat from a file.

        Parameters
        ----------
        filename : str, optional
            Filename to load the dark frame from. If not specified, uses the dark file path from the configuration.
        """
        # If no file given, first try dark file
        try:
            if filename == "":
                filename = self.flat_file
            if filename == "":
                flat = np.zeros_like(self.flat)
                self.logger.info("No flat file configured; using zeros")
            else:
                if ".txt" in filename:
                    flat = np.genfromtxt(filename)
                elif ".npy" in filename:
                    flat = np.load(filename)
                else:
                    raise ValueError(f"Unsupported flat file format: {filename}")
                self.logger.info("Loaded flat from %s", filename)
            self.set_flat(flat)
        except Exception:
            self.logger.exception("Failed to load flat from %s", filename or self.flat_file)
            raise
        return

    def set_layout(self, layout):
        """
        Set the layout of the actuators.

        Parameters
        ----------
        layout : numpy.ndarray or None
            Layout of the actuators. Is converted to boolean if not already.
        """
        try:
            self.layout = layout
            if isinstance(self.layout, np.ndarray):
                self.layout = self.layout > 0
                self.correction_vector_2d = create_stream(
                    self.output_stream_name("wfc_2d"),
                    self.layout.shape,
                    np.float32,
                    gpu_device=self.gpu_device,
                )
                self.register_output_stream("wfc_2d", self.correction_vector_2d)
                self.write_stream("wfc_2d", np.zeros(self.layout.shape, dtype=np.float32))
                self.correction_vector_2d_template = self.read_stream("wfc_2d", block=False)

                self.index_map = np.zeros(self.layout.shape, dtype=int)
                self.index_map[self.layout > 0] = np.arange(np.sum(self.layout)).astype(int) + 1
                self.logger.info("Configured 2D correction layout shape=%s", self.layout.shape)
                if (
                    getattr(self, "m2c_source", None) == "basis"
                    and not self.basis_conf.positions_file
                    and self.basis_actuator_positions() is None
                ):
                    # The basis was built before the real layout was known.
                    self.build_basis_m2c()
            else:
                self.logger.info("Cleared 2D correction layout")
        except Exception:
            self.logger.exception("Failed to set wavefront corrector layout")
            raise

        return

    def deactivate_actuators(self, actuators):
        """
        Deactivate specified actuators. Actuators are assumed to be floating

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to deactivate.
        """

        try:
            if hasattr(actuators, "__len__") and len(actuators) < 1:
                raise Exception("You have provided no actuators")
            if not hasattr(actuators, "__len__"):
                raise Exception("Actuators given as wrong type, please provide array or list")

            if isinstance(self.layout, np.ndarray):
                if len(self.layout.shape) != 2:
                    raise Exception(
                        "Layout must be 2 dimensions to float actuators. To remove dead actuators, remove them from the M2C. OR set the layout to be 2D and the floating_influence_radius to a 0"
                    )
                act_to_float_mask = np.zeros_like(self.index_map)
                for act in actuators:
                    act_to_float_mask[np.where(self.index_map == act + 1)] = 1
                    self.actuator_status[act] = False

                for act in actuators:
                    i, j = np.where(self.index_map == act + 1)
                    inlfluence_map = gaussian_2d_grid(
                        i, j, self.floating_influence_radius, self.layout.shape[0]
                    )
                    inlfluence_map *= self.layout * (1 - act_to_float_mask)
                    inlfluence_map /= np.sum(inlfluence_map)
                    inlfluence_map[inlfluence_map < np.max(inlfluence_map) / 10] = 0
                    self.float_matrix[act] = inlfluence_map[self.layout > 0]

                m2c_source = getattr(self, "m2c_source", "custom")
                self.set_m2c(self.M2C)
                self.m2c_source = m2c_source
                self.logger.info("Deactivated actuators %s", actuators)
            else:
                logger.warning("No layout set for DM")
        except Exception:
            self.logger.exception("Failed to deactivate actuators %s", actuators)
            raise

        return

    def reactivate_actuators(self, actuators):
        """
        Reactivate specified actuators.

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to reactivate.
        """
        try:
            for act in actuators:
                self.actuator_status[act] = True
            self.float_matrix = np.eye(self.num_actuators, dtype=self.flat.dtype)
            acts_to_deactivate = [
                i for i in range(self.num_actuators) if not self.actuator_status[i]
            ]
            if len(acts_to_deactivate) > 0:
                self.deactivate_actuators(acts_to_deactivate)
            self.logger.info("Reactivated actuators %s", actuators)
        except Exception:
            self.logger.exception("Failed to reactivate actuators %s", actuators)
            raise
        return

    def set_m2c(self, M2C):
        """
        Set the mode-to-command matrix. This is the basis for correction.

        Parameters
        ----------
        M2C : numpy.ndarray or None
            Mode-to-command matrix to set. Axes are [num_actuators, num_modes]
        """
        try:
            if not isinstance(M2C, np.ndarray):
                self.M2C = np.eye(self.num_actuators)[:, : self.num_modes]
            else:
                self.M2C = M2C

            self.M2C = self.M2C.astype(self.flat.dtype)
            # Callers that know the origin (file, basis) overwrite this.
            self.m2c_source = "custom" if isinstance(M2C, np.ndarray) else "identity"

            self.f_M2C = self.float_matrix @ self.M2C

            self.C2M = np.linalg.pinv(self.M2C)
            self.num_modes = self.M2C.shape[1]
            self.current_correction = np.zeros(self.num_modes, dtype=self.flat.dtype)
            self.flat_modal = self.C2M @ self.flat
            self.logger.info("Configured M2C matrix shape=%s", self.M2C.shape)
        except Exception:
            self.logger.exception("Failed to configure M2C matrix")
            raise

    def set_delay(self, delay):
        """
        Sets an artificial frame delay. Used for testing, nominally the delay should always be zero.

        Parameters
        ----------
        delay : int
            Frame delay to set.
        """
        try:
            self.frame_delay = delay
            self.shape_buffer = np.zeros(
                (self.frame_delay + 1, *self.current_shape.shape), dtype=self.current_shape.dtype
            )
            for i in range(self.shape_buffer.shape[0]):
                self.shape_buffer[i] = self.flat.copy()
            self.logger.info("Set artificial frame delay to %s", delay)
        except Exception:
            self.logger.exception("Failed to set frame delay to %s", delay)
            raise

        return

    def read_m2c(self, filename=""):
        """
        Load the mode-to-command matrix.

        Precedence: an explicit ``filename``, then the configured ``m2c_file``,
        then the configured ``basis`` (built with aobasis), then the identity.

        Parameters
        ----------
        filename : str, optional
            File to read the mode-to-command matrix from. If not specified, uses the configured m2c_file.
        """
        try:
            if filename == "":
                filename = self.m2c_file

            if ".dat" in filename:
                M2C = np.fromfile(filename, dtype=np.float64).reshape(
                    self.num_actuators, self.num_modes
                )
            elif ".npy" in filename:
                M2C = np.load(filename)
            elif getattr(self, "basis_conf", None) is not None:
                self.build_basis_m2c()
                return
            else:
                self.set_m2c(None)
                self.m2c_source = "identity"
                self.logger.info("No M2C file or basis configured; using identity basis")
                return

            self.set_m2c(M2C)
            self.m2c_source = "file"
            self.logger.info("Loaded M2C matrix from %s", filename)
        except Exception:
            self.logger.exception("Failed to read M2C matrix from %s", filename or self.m2c_file)
            raise
        return

    def basis_actuator_positions(self):
        """
        Return ``(num_actuators, 2)`` actuator positions in metres, or ``None``.

        Adapters that know their physical actuator coordinates override this.
        ``None`` makes :meth:`build_basis_m2c` derive positions from
        :attr:`layout` (see :func:`pyrtc.modal_basis.actuator_positions_from_layout`).
        """
        return None

    def build_basis_m2c(self, basis=None):
        """
        Build ``M2C`` with aobasis from a basis config and apply it.

        Positions come from ``basis.positions_file``, then
        :meth:`basis_actuator_positions`, then the 2D :attr:`layout`. Without a
        layout yet, the default circular layout for ``num_actuators`` is used
        and the basis is rebuilt when :meth:`set_layout` provides the real one.

        Parameters
        ----------
        basis : dict or pyrtc.modal_basis.BasisConfig, optional
            Basis to build. Defaults to the configured ``basis`` section. A
            new basis given here replaces the configured one.

        Returns
        -------
        numpy.ndarray
            The new ``M2C`` matrix, ``(num_actuators, num_modes)``.
        """
        try:
            if basis is not None:
                if not isinstance(basis, BasisConfig):
                    basis = parse_basis_config(basis, num_modes=self.num_modes)
                self.basis_conf = basis
                self._basis_layout = None
            if self.basis_conf is None:
                raise ValueError("No basis configured")

            positions = None
            layout = None
            if not self.basis_conf.positions_file:
                positions = self.basis_actuator_positions()
            if positions is None and not self.basis_conf.positions_file:
                layout = self.layout
                if not isinstance(layout, np.ndarray) or layout.ndim != 2:
                    layout = default_actuator_layout(self.num_actuators)
                if int(np.count_nonzero(layout)) != int(self.num_actuators):
                    raise ValueError(
                        f"basis: layout has {int(np.count_nonzero(layout))} active actuators "
                        f"but num_actuators is {self.num_actuators}"
                    )
                cached = getattr(self, "_basis_layout", None)
                if cached is not None and np.array_equal(layout, cached):
                    if getattr(self, "m2c_source", None) == "basis":
                        return self.M2C
                self._basis_layout = np.array(layout, dtype=bool)

            M2C = build_m2c(
                self.basis_conf,
                num_modes=self.num_modes,
                layout=layout,
                positions=positions,
                logger=self.logger,
            )
            if M2C.shape[0] != self.num_actuators:
                raise ValueError(
                    f"basis: {M2C.shape[0]} actuator positions but num_actuators is "
                    f"{self.num_actuators}"
                )
            self.set_m2c(M2C)
            self.m2c_source = "basis"
            return self.M2C
        except Exception:
            self.logger.exception("Failed to build M2C from basis config")
            raise

    def send_to_hardware(self):
        """
        Send the current correction to the hardware. Nominally, this function is overwritten by the
        child hardware class and registered to the real-time loop from the config.
        """
        # Read a new modal correction in M2C basis
        self.current_correction = self.read_stream("wfc", out=self._wfc_buffer)
        # If we added a frame delay
        if self.frame_delay > 0:
            # Roll back shape buffer by 1
            self.shape_buffer[:-1] = self.shape_buffer[1:]
            # Compute a new shape in zonal basis
            self.shape_buffer[-1] = ModaltoZonalWithFlat(
                self.current_correction, self.f_M2C, self.flat
            )
            # Set the current shape
            self.current_shape = self.shape_buffer[0]
        else:
            self.current_shape = ModaltoZonalWithFlat(
                self.current_correction, self.f_M2C, self.flat
            )

        if self.command_cap is not None:
            self.current_shape = np.clip(self.current_shape, -self.command_cap, self.command_cap)

        # If we have a 2D SHM instance, update it
        if self.correction_vector_2d is not None:
            self.correction_vector_2d_template.fill(0)
            self.correction_vector_2d_template[self.layout] = self.current_shape - self.flat
            self.write_stream("wfc_2d", self.correction_vector_2d_template)
        # Overwrite with hardware instructions after this to send to hardware
        return

    def read(self, block=False):
        """
        Read the current correction vector.

        Returns
        -------
        numpy.ndarray
            Current correction vector.
        """
        if block:
            return self.read_stream("wfc")
        return self.read_stream("wfc", block=False)

    def write(self, correction):
        """
        Write a new correction.

        Parameters
        ----------
        correction : numpy.ndarray
            Correction vector to write.
        """
        self.current_correction = correction
        # We assume that send_to_hardware is registered to the real-time loop
        # And that the WFC is running (i.e. start has been called)
        self.write_stream("wfc", self.current_correction)
        return

    def flatten(self):
        """
        Flatten the wavefront corrector.
        """
        # Sending a zero correction will be the flat since the correction
        # is always assumed to be on top of the flat.
        try:
            self.write(np.zeros_like(self.current_correction))
            self.logger.info("Flattened wavefront corrector")
        except Exception:
            self.logger.exception("Failed to flatten wavefront corrector")
            raise
        return

    def push(self, mode, amp):
        """
        Push a specific mode with a given amplitude.

        Parameters
        ----------
        mode : int
            Mode index to push.
        amp : float
            Amplitude to push the mode with.
        """
        try:
            corr = np.zeros_like(self.current_correction)
            corr[int(mode)] = float(amp)
            self.write(corr)
            self.logger.info("Pushed mode %s with amplitude %s", mode, amp)
        except Exception:
            self.logger.exception("Failed to push mode %s with amplitude %s", mode, amp)
            raise
        return

    def save_shape(self, filename=""):
        """
        Save the current shape to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the shape to. If not specified, uses the configured save_file.
        """
        try:
            if filename == "":
                filename = self.save_file
            if filename == "":
                raise ValueError("No output filename provided for shape save")
            np.save(filename, self.current_shape)
            self.logger.info("Saved current shape to %s", filename)
        except Exception:
            self.logger.exception("Failed to save current shape to %s", filename or self.save_file)
            raise
        return

    def plot(self, add_flat=False):
        """
        Plot the current correction.

        Returns the figure (not shown); call ``plt.show()`` to display it.

        Parameters
        ----------
        add_flat : bool, optional
            If True, adds the flat shape to the current correction before plotting. Default is False.
        """
        cur_correction = self.read()
        if add_flat:
            cur_correction += self.flat_modal

        if isinstance(self.layout, np.ndarray):
            new_shape = np.zeros(self.layout.shape)
            new_shape[self.layout] = self.M2C @ cur_correction
        else:
            new_shape = cur_correction

        plt = pyplot()
        fig, ax = plt.subplots()
        if len(new_shape.shape) == 1:
            ax.plot(new_shape)
        else:
            im = ax.imshow(new_shape, cmap="inferno", aspect="auto", origin="lower")
            fig.colorbar(im, ax=ax)
        return fig


if __name__ == "__main__":
    launch_component(WavefrontCorrector, "wfc", start=True)
