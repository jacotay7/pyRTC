import json

from benchmarks import core_compute_bench


REQUIRED_CPU_KERNELS = [
    "wavefront_sensor.downsample_int32_image_jit",
    "wavefront_sensor.rotate_image_jit",
    "wavefront_corrector.ModaltoZonalWithFlat",
    "loop.leakyIntegratorNumba",
    "slopes.computeSlopesPYWFSOptimNumba",
    "slopes.computeSlopesSHWFSOptimNumba",
]


def test_run_core_compute_benchmarks_cpu_quick_schema():
    report = core_compute_bench.run_core_compute_benchmarks(
        iterations=1,
        warmup=1,
        include_gpu=False,
        quick=True,
        system_sizes=[10, 20, 60],
    )

    assert "meta" in report
    assert "profiles" in report
    for profile in ["10x10", "20x20", "60x60"]:
        assert profile in report["profiles"]
        for kernel_name in REQUIRED_CPU_KERNELS:
            assert kernel_name in report["profiles"][profile]
            stats = report["profiles"][profile][kernel_name]
            assert stats["mean_s"] >= 0
            assert stats["p99_hz"] >= 0


def test_core_compute_bench_main_writes_json(tmp_path):
    output = tmp_path / "core_bench.json"
    code = core_compute_bench.main(
        [
            "--quick",
            "--cpu-only",
            "--iterations",
            "1",
            "--warmup",
            "1",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "profiles" in payload
    assert "meta" in payload
