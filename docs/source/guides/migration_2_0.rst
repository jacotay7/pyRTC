Migrating to pyrtc 2.0
======================

pyrtc 2.0 changes two conventions that images, slopes and saved calibrations
depend on. Both bring pyrtc in line with the AO stack's shared conventions
(aocore ``CONVENTIONS.md``), so slopes and images mean the same in pyrtc as in
its sibling packages, AOTPy exports and milk/CACAO.

What changed
------------

**Image axes (#162).** Image streams (``wfs_raw``, ``wfs``, ``psf_short``,
``psf_long``) are ``(height, width)`` arrays indexed ``[y, x]``: rows are the
camera's y axis and columns its x axis. pyrtc 1.x declared them
``(width, height)``.

- Camera adapters publish frames as the SDK returns them. The GenICam and
  Micro-Manager adapters no longer transpose; the XIMEA and Spinnaker
  adapters, which did not transpose, now also work with non-square ROIs
  (#130).
- Slope x and y follow the camera's columns and rows. Through a transposing
  1.x adapter, pyrtc's "x" slope was the camera's y, and its sub-apertures were
  ordered column-major in camera terms.
- PYWFS ``pupils`` entries are ``"x,y"`` = column,row of the image. pyrtc 1.x
  read them as row,column of its stream.
- A non-square SHWFS image gets ``min(height, width) // round(sub_ap_spacing)``
  sub-apertures per side (1.x used the first stream axis).
- The ISIO bridge reverses the axes, so milk sees a pyrtc image with
  ``size = [width, height]``, as before.

**SHWFS pixel centres (#163).** Pixel ``k`` of a sub-aperture sits at
``k - (n - 1) / 2``, not ``k - n // 2``. A spot on the optical axis of an
even-sized sub-aperture (between its two middle pixels) now reads 0, where
1.x read -0.5 px. Raw slopes of even sub-apertures move by +0.5 px on both
axes; slopes measured against reference slopes do not change. Odd sizes are
unaffected.

**Calibration files.** Darks, model PSFs, SHWFS reference images, valid
sub-aperture masks, reference slopes and interaction matrices are saved as
versioned files: ``.npz`` archives with the array under ``data`` and a
``pyrtc_calibration`` record (format version, kind). They keep the file name
the config gives (``np.load`` detects the format from the content), and
:func:`pyrtc.calibration.load_calibration` reads them. A plain ``.npy`` file
(what 1.x wrote) is a *legacy* file, and components refuse it unless told how
it was made. Corrector files (flat, M2C, layouts) do not depend on the camera
and stay plain ``.npy``.

Which case are you?
-------------------

What a 1.x calibration means in 2.0 depends on whether the 1.x camera adapter
transposed frames:

``yx``: frames published unchanged
  XIMEA, Spinnaker, and the simulators (synthetic, OOPAO, HCIPy, SPECULA).
  Their 1.x stream was already the camera's ``[y, x]`` array (square ROIs
  only), so images, masks and interaction matrices are unchanged. SHWFS
  reference slopes of even sub-apertures move by +0.5 px. Explicit PYWFS
  ``pupils`` strings must be swapped (``"a,b"`` becomes ``"b,a"``) to keep
  the same pupils; also reorder them as described in
  :doc:`../components/slopes` if you want ``sx`` to compare columns.
``xy``: frames transposed into ``(width, height)``
  GenICam, Micro-Manager, or an adapter of your own that applied ``frame.T``
  (as the 1.x docs advised). Images transpose; SHWFS slope maps and valid
  masks swap their x and y halves and transpose each; reference slopes also
  move by +0.5 px; interaction-matrix rows are reordered. PYWFS ``pupils``
  strings stay as they are (1.x read them as camera x,y through the
  transpose).
``as_is``
  The file was written by hand or by another tool and already follows the
  2.0 conventions.

Converting calibration files
----------------------------

Set ``legacy_calibration`` in each section that loads 1.x files, and pyrtc
converts them on load, with a warning:

.. code-block:: yaml

  wfs:
    dark_file: calib/dark.npy
    legacy_calibration: xy          # or yx / as_is
  slopes:
    valid_sub_aps_file: calib/valid_sub_aps.npy
    ref_slopes_file: calib/ref.npy
    legacy_calibration: xy
  loop:
    im_file: calib/im.npy
    legacy_calibration: yx          # 'xy' interaction matrices: see below
  psf:
    dark_file: calib/psf_dark.npy
    model_file: calib/model_psf.npy
    legacy_calibration: xy

Then save each file again (``save_dark()``, ``save_valid_sub_aps()``,
``save_ref_slopes()``, ``save_im()``, ...) and remove ``legacy_calibration``.
Saved files are 2.0 files and load without it.

Every conversion is exact. Two cases cannot be converted on load:

- An **interaction matrix from an** ``xy`` **adapter** needs the 1.x valid
  sub-aperture mask to reorder its rows. Convert it offline::

    pyrtc-migrate-calibration interaction_matrix calib/im.npy calib/im_v2.npz \
        --legacy-frame xy --wfs-type shwfs --valid-sub-aps calib/valid_sub_aps.npy

  (for a PYWFS: ``--wfs-type pywfs``, the 1.x PYWFS valid mask, and
  ``--default-pupils`` if its config had no ``pupils``).
- **WCoG reference slopes taken without a reference image** on even
  sub-apertures: the Gaussian weight was centred half a pixel off, so the
  old reference is not exactly a shifted new one. Take reference slopes again.

``pyrtc-migrate-calibration`` converts any kind of file
(``wfs_dark``, ``psf_dark``, ``psf_model``, ``reference_image``,
``valid_sub_aps``, ``ref_slopes``, ``interaction_matrix``); see
``pyrtc-migrate-calibration --help``. In Python,
:func:`pyrtc.calibration.migrate_calibration_file` does the same.

Re-measuring instead
--------------------

Re-measuring is always correct, and for darks and reference slopes it is
quick:

- WFS and science-camera darks: ``take_dark()``, then ``save_dark()``.
- Model PSF: ``take_model_psf()``, then ``save_model_psf()``.
- SHWFS reference image: ``take_reference_image()``, then
  ``save_reference_image()``.
- Reference slopes: ``take_ref_slopes()``, then ``save_ref_slopes()``.
- Valid sub-aperture mask: build it again and ``save_valid_sub_aps()``.
- Interaction matrix: ``compute_im()``, then ``save_im()``.

Other things to check
---------------------

- Code that reads ``wfs``/``psf`` streams and assumed ``(width, height)``
  (custom viewers, analysis scripts, reconstructor training data) now gets
  ``(height, width)`` arrays.
- A :doc:`TorchImageReconstructor <../components/image_reconstructor>`
  model trained on frames from an ``xy`` adapter sees transposed images:
  retrain it, or wrap it to transpose its input.
- Your own camera adapters should publish frames as ``(height, width)``
  without transposing them.
- Scripts that ``np.load`` a calibration file written by 2.0 get an
  ``NpzFile``; read ``["data"]``, or use
  :func:`pyrtc.calibration.load_calibration`.
