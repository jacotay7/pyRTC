"""Check the focal-plane WFS research environment and measure its GPU headroom.

Runs three stages:

1. imports and a minimal run of every simulation tool the project uses;
2. batch-1 latency of a CNN reconstructor (eager and CUDA graph) and of a dense
   linear reconstructor, which sets the real-time compute budget;
3. batched focal-plane image simulation throughput, which sets how fast
   training data can be generated.

Usage::

    python research/fpwfs/tools/env_check.py [--device cuda:0] [--skip-bench]

Pin to quiet cores (``taskset -c ...``) on a shared host; GPU numbers are the
ones to trust, the CPU launch overhead is part of them.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def check(name, fn):
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn()
        print(f"OK   {name:10s} {time.perf_counter() - t0:6.2f}s  {result}")
        return True
    except Exception as exc:  # report every failure, keep going
        print(f"FAIL {name:10s} {type(exc).__name__}: {exc}")
        return False


def _torch():
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return f"{torch.__version__} {names}"


def _cupy():
    import cupy as cp

    return f"{cp.__version__} sum={float((cp.arange(10) ** 2).sum())}"


def _hcipy():
    import hcipy as hc

    grid = hc.make_pupil_grid(128, 8)
    aperture = hc.evaluate_supersampled(hc.make_vlt_aperture(), grid, 4)
    focal = hc.make_focal_grid(4, 16, spatial_resolution=1.6e-6 / 8)
    img = hc.FraunhoferPropagator(grid, focal)(hc.Wavefront(aperture, 1.6e-6)).power
    return f"{hc.__version__} VLT PSF peak fraction {img.max() / img.sum():.4f}"


def _pyturb():
    import pyturb

    atm = pyturb.Atmosphere.from_profile("paranal-median", seeing=0.8, diameter=8.0, n=128)
    opd = np.asarray(atm.opd(0.0))
    return f"{pyturb.__version__} paranal-median OPD rms {opd.std() * 1e9:.0f} nm"


def _getframes():
    import cupy as cp
    import getframes as gf

    cam = gf.Camera.from_preset("first_light_imaging_cred_one", device="gpu", precision="float32")
    frame = cam.expose(cp.full(cam.resolution, 1e5, dtype=cp.float32), exposure=1e-3, seed=0)
    return f"{gf.__version__} C-RED One {cam.resolution} -> {type(frame.data).__module__}"


def _makewfs():
    import pathlib

    import makewfs

    root = pathlib.Path(makewfs.__file__).resolve().parents[2]
    cfg = root / "examples" / "configs" / "shack_hartmann_minimal.toml"
    wfs = makewfs.WavefrontSensor.from_toml(cfg)
    frame = wfs.expose(np.zeros(wfs.config.input.shape), seed=0)
    return f"{makewfs.__version__} SH frame {np.shape(frame.data)}"


def _specula():
    import specula

    specula.init(0, precision=1)
    from specula import xp

    return f"{specula.__name__} backend={xp.__name__}"


def _oopao():
    from OOPAO.Atmosphere import Atmosphere
    from OOPAO.Source import Source
    from OOPAO.Telescope import Telescope

    tel = Telescope(resolution=80, diameter=8, samplingTime=1e-3, centralObstruction=0.14)
    ngs = Source("H", 8)
    ngs * tel
    atm = Atmosphere(
        telescope=tel,
        r0=0.15,
        L0=25,
        windSpeed=[10],
        fractionalR0=[1],
        windDirection=[0],
        altitude=[0],
    )
    atm.initializeAtmosphere(tel)
    return f"OPD rms {np.std(atm.OPD[tel.pupil > 0]) * 1e9:.0f} nm"


def _aobasis():
    import aobasis

    return aobasis.__version__


def _pyrtc():
    import pyrtc

    return getattr(pyrtc, "__version__", "imported")


def bench(device: str, batch: int) -> None:
    import torch
    import torch.nn as nn

    torch.backends.cudnn.benchmark = True
    dev = torch.device(device)
    torch.cuda.set_device(dev)  # graph capture uses the current device
    print(f"\ndevice: {torch.cuda.get_device_name(dev)}")

    def timed(fn, n=2000, warm=200):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(dev)
        ts = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            ts[i] = time.perf_counter() - t0
        return np.median(ts) * 1e6, np.percentile(ts, 99) * 1e6

    def cnn(cin, nmodes, w=32):
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 2, 1),
                nn.BatchNorm2d(b),
                nn.GELU(),
                nn.Conv2d(b, b, 3, 1, 1),
                nn.BatchNorm2d(b),
                nn.GELU(),
            )

        return nn.Sequential(
            blk(cin, w),
            blk(w, 2 * w),
            blk(2 * w, 4 * w),
            blk(4 * w, 8 * w),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(8 * w * 16, 1024),
            nn.GELU(),
            nn.Linear(1024, nmodes),
        )

    def graphed(model, x):
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(5):
                model(x)
        torch.cuda.current_stream(dev).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(x)
        return graph.replay

    print("batch-1 reconstructor latency, us (median / p99)")
    for npix, cin, nmodes in [(96, 4, 400), (128, 4, 800), (160, 4, 1200)]:
        for dtype in (torch.float32, torch.float16):
            model = cnn(cin, nmodes).to(dev, dtype).eval()
            x = torch.randn(1, cin, npix, npix, device=dev, dtype=dtype)
            with torch.inference_mode():
                eager = timed(lambda: model(x), n=500)
                graph = timed(graphed(model, x))
            mparams = sum(p.numel() for p in model.parameters()) / 1e6
            print(
                f"  CNN {cin}x{npix}^2 -> {nmodes:4d} modes {str(dtype)[6:]:7s} {mparams:4.1f}M: "
                f"eager {eager[0]:5.0f}/{eager[1]:5.0f}  cuda-graph {graph[0]:5.0f}/{graph[1]:5.0f}"
            )
        rmat = torch.randn(nmodes, cin * npix * npix, device=dev, dtype=torch.float16)
        vec = torch.randn(cin * npix * npix, device=dev, dtype=torch.float16)
        dense = timed(lambda: torch.mv(rmat, vec))[0]
        print(f"  dense linear {cin * npix * npix:6d} -> {nmodes:4d} fp16: {dense:5.0f} us")

    n, pad = 128, 256
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n, device=dev), torch.linspace(-1, 1, n, device=dev), indexing="ij"
    )
    r = torch.hypot(xx, yy)
    pupil = ((r <= 1) & (r >= 0.14)).float()
    phase = torch.randn(batch, n, n, device=dev) * pupil

    def simulate():
        field = torch.zeros(batch, 4, pad, pad, device=dev, dtype=torch.complex64)
        for k in range(4):  # four diversity states per sample
            field[:, k, :n, :n] = pupil * torch.exp(1j * (phase + 0.3 * k * r**2))
        img = torch.fft.fftshift(torch.fft.fft2(field).abs() ** 2, dim=(-2, -1))
        return img[..., 64:192, 64:192]

    med, _ = timed(simulate, n=30, warm=5)
    print(
        f"focal-plane sim (128 px pupil, Nyquist, 128^2 crop, 4 frames/sample): "
        f"{batch} samples in {med / 1e3:.1f} ms -> {batch / med * 1e6:,.0f} samples/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=256, help="sim batch (A400 4 GB: <=256)")
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    checks = [
        ("torch", _torch),
        ("cupy", _cupy),
        ("hcipy", _hcipy),
        ("pyturb", _pyturb),
        ("getframes", _getframes),
        ("makewfs", _makewfs),
        ("specula", _specula),
        ("oopao", _oopao),
        ("aobasis", _aobasis),
        ("pyrtc", _pyrtc),
    ]
    ok = all([check(name, fn) for name, fn in checks])
    if not args.skip_bench:
        bench(args.device, args.batch)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
