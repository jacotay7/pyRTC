import json
from pathlib import Path

from benchmarks import ao_loop_bench, readme_benchmark_table


def test_run_ao_loop_benchmarks_cpu_quick_schema():
    report = ao_loop_bench.run_ao_loop_benchmarks(
        iterations=1,
        warmup=1,
        include_gpu=False,
        system_sizes=[10, 20, 60],
    )

    assert report["meta"]["benchmark_type"] == "synthetic_closed_loop"
    for sensor in ["pywfs", "shwfs"]:
        assert sensor in report["results"]
        for profile in ["10x10", "20x20", "60x60"]:
            assert profile in report["results"][sensor]
            stats = report["results"][sensor][profile]["cpu"]
            assert stats["mean_s"] >= 0
            assert stats["p99_hz"] >= 0


def test_ao_loop_bench_main_writes_json(tmp_path):
    output = tmp_path / "ao_loop_bench.json"
    code = ao_loop_bench.main(
        [
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
    assert payload["meta"]["benchmark_type"] == "synthetic_closed_loop"
    assert "results" in payload


def test_ao_loop_bench_main_without_output_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    code = ao_loop_bench.main(
        [
            "--cpu-only",
            "--iterations",
            "1",
            "--warmup",
            "1",
        ]
    )

    assert code == 0
    assert not Path("benchmarks/ao_loop_bench_report.json").exists()


def test_readme_table_supports_closed_loop_reports():
    report = {
        "meta": {
            "benchmark_type": "synthetic_closed_loop",
            "system_sizes": [10, 20, 60],
            "system_info": {
                "cpu_model": "cpu",
                "cpu_count": 8,
                "gpu": {"device_name": "gpu"},
            },
        },
        "results": {
            "pywfs": {
                "10x10": {"cpu": {"p99_hz": 1000.0, "p99_s": 0.001}, "gpu": {"p99_hz": 2000.0, "p99_s": 0.0005}},
                "20x20": {"cpu": {"p99_hz": 800.0, "p99_s": 0.00125}, "gpu": {"p99_hz": 1600.0, "p99_s": 0.000625}},
                "60x60": {"cpu": {"p99_hz": 100.0, "p99_s": 0.01}, "gpu": {"p99_hz": 400.0, "p99_s": 0.0025}},
            },
            "shwfs": {
                "10x10": {"cpu": {"p99_hz": 1200.0, "p99_s": 0.000833}, "gpu": {"p99_hz": 1800.0, "p99_s": 0.000556}},
                "20x20": {"cpu": {"p99_hz": 900.0, "p99_s": 0.001111}, "gpu": {"p99_hz": 1400.0, "p99_s": 0.000714}},
                "60x60": {"cpu": {"p99_hz": 120.0, "p99_s": 0.008333}, "gpu": {"p99_hz": 450.0, "p99_s": 0.002222}},
            },
        },
    }

    markdown = readme_benchmark_table.build_markdown(report)
    assert "Synthetic AO Loop Benchmarks" in markdown
    assert "PYWFS full loop" in markdown
    assert "SHWFS full loop" in markdown


def test_ao_loop_summary_table_contains_compact_headers():
    report = {
        "results": {
            "pywfs": {
                "10x10": {
                    "cpu": {"p99_hz": 1000.0, "p99_s": 0.001},
                    "gpu": {"p99_hz": 2000.0, "p99_s": 0.0005},
                }
            }
        }
    }

    table = ao_loop_bench._build_summary_table(report)
    assert "Sensor" in table
    assert "Size" in table
    assert "PYWFS" in table
    assert "10x10" in table