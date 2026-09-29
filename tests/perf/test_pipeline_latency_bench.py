"""Smoke tests for the end-to-end pipeline latency benchmark.

These launch the real synthetic SHWFS system with tiny sample counts. Every
run gets a private stream-name prefix, so they cannot collide with another
pyrtc system (or test) using the canonical ``wfs``/``signal``/``wfc`` names.
"""

import json

import pyshmem
import yaml

from benchmarks import pipeline_latency_bench as bench

CANONICAL = {"wfs", "wfs_raw", "signal", "signal_2d", "wfc", "wfc_2d", "psf_short", "psf_long"}


def _leftover_streams(prefix):
    return [name for name in pyshmem.list_streams() if name.startswith(prefix)]


def test_benchmark_config_uses_only_private_stream_names(tmp_path):
    config_path, names = bench.build_benchmark_config("bench_test_prefix", tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert set(bench.LATENCY_PATH) <= set(names)
    for section in config.values():
        if not isinstance(section, dict) or "class_name" not in section:
            continue
        for direction in ("input_streams", "output_streams"):
            for stream_name in (section.get(direction) or {}).values():
                assert stream_name.startswith("bench_test_prefix_")
                assert stream_name not in CANONICAL
    assert config["wfs"]["input_streams"]["wfc"] == names["wfc"]
    assert (tmp_path / "synthetic_im.npy").exists()


def test_soft_pipeline_latency_smoke():
    report = bench.run_pipeline_benchmarks(
        modes=("soft",),
        notify_settings=(False, True),
        samples=8,
        settle_seconds=0.5,
        frame_rate_hz=500.0,
        include_psf=False,
        timeout_seconds=60.0,
    )

    assert report["meta"]["benchmark_type"] == "pipeline_latency"
    assert [row["notify"] for row in report["results"]] == [False, True]
    for row in report["results"]:
        assert row["mode"] == "soft-rtc"
        assert row["notify_flag_on_streams"] is row["notify"]
        assert row["total"]["count"] >= 1
        assert row["total"]["mean_us"] > 0
        assert [(seg["source"], seg["target"]) for seg in row["segments"]] == [
            ("wfs", "signal"),
            ("signal", "wfc"),
        ]
        assert _leftover_streams(row["stream_prefix"]) == []
    assert [(row["mode"], row["notify"], row["runs"]) for row in report["summary"]] == [
        ("soft-rtc", False, 1),
        ("soft-rtc", True, 1),
    ]
    assert "soft-rtc" in bench.format_report(report)


def test_summarize_repeats_takes_the_median():
    def _stats(value):
        return {key: value for key in bench._STAT_KEYS} | {"count": 10}

    rows = [
        {
            "mode": "hard-rtc",
            "notify": True,
            "stream_path": ["wfs", "signal"],
            "total": _stats(value),
            "segments": [{"source": "wfs", "target": "signal", **_stats(value)}],
        }
        for value in (1.0, 5.0, 3.0)
    ]
    (summary,) = bench.summarize_repeats(rows)
    assert summary["runs"] == 3
    assert summary["total"]["p50_us"] == 3.0
    assert summary["total"]["count"] == 30
    assert summary["segments"][0]["mean_us"] == 3.0


def test_pipeline_latency_main_writes_json(tmp_path):
    output = tmp_path / "pipeline.json"
    code = bench.main(
        [
            "--modes",
            "soft",
            "--notify",
            "on",
            "--samples",
            "4",
            "--settle",
            "0.5",
            "--frame-rate-hz",
            "500",
            "--no-psf",
            "--timeout",
            "60",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"][0]["notify"] is True
    assert payload["results"][0]["total"]["p99_us"] >= payload["results"][0]["total"]["p50_us"]
