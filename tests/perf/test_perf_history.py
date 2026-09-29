"""Benchmark history tool (#65)."""

import io
import json
import zipfile

import pytest

from benchmarks import perf_history


def _report(scale=1.0, timestamp=0.0):
    return {
        "timestamp_unix": timestamp,
        "core_compute": {
            "profiles": {
                "10x10": {
                    "loop.leaky": {"median_s": 2e-6 * scale, "mean_s": 9.0},
                    "slopes.shwfs": {"mean_s": 4e-6},
                }
            }
        },
        "platform": "linux",
    }


def test_flatten_prefers_median_and_joins_paths():
    metrics = perf_history.flatten_report(_report())
    assert metrics == {
        "core_compute/profiles/10x10/loop.leaky": 2e-6,
        "core_compute/profiles/10x10/slopes.shwfs": 4e-6,
    }


def test_flatten_reads_pipeline_latency_summaries():
    report = {
        "summary": [
            {
                "mode": "soft-rtc",
                "notify": True,
                "total": {"p50_us": 300.0, "mean_us": 9.0},
                "segments": [{"source": "wfs", "target": "signal", "p50_us": 120.0}],
            },
        ],
        "results": [],
    }
    metrics = perf_history.flatten_report(report)
    assert metrics == {
        "summary/soft-rtc,notify/total": pytest.approx(300e-6),
        "summary/soft-rtc,notify/segments/wfs->signal": pytest.approx(120e-6),
    }


def test_compare_flags_the_newest_regression():
    history = [perf_history.flatten_report(_report(s)) for s in (1.0, 1.1, 0.9, 1.0, 3.0)]
    rows = perf_history.compare(history, window=10)
    worst = rows[0]
    assert worst["metric"].endswith("loop.leaky")
    assert worst["ratio"] == pytest.approx(3.0)
    assert worst["runs"] == 4
    assert rows[-1]["ratio"] == pytest.approx(1.0)
    table = perf_history.format_table(rows, max_ratio=1.5)
    assert "REGRESSION" in table.splitlines()[2] and "REGRESSION" not in table.splitlines()[3]


def test_directory_source_orders_by_timestamp(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_report(2.0, timestamp=20.0)))
    (tmp_path / "b.json").write_text(json.dumps(_report(1.0, timestamp=10.0)))
    runs = perf_history.load_directory(tmp_path)
    assert [name for name, _ in runs] == ["b.json", "a.json"]
    assert perf_history.main(["--from-dir", str(tmp_path), "--max-ratio", "1.5"]) == 1
    assert perf_history.main(["--from-dir", str(tmp_path), "--max-ratio", "2.5"]) == 0


def test_github_source_reads_artifacts_oldest_first():
    def artifact_zip(report):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("perf_smoke_report.json", json.dumps(report))
        return buffer.getvalue()

    runs = {
        "runs": {
            "workflow_runs": [
                {"id": 2, "created_at": "2026-09-29T02:00:00Z", "head_sha": "bbbbbbbb"},
                {"id": 1, "created_at": "2026-09-28T02:00:00Z", "head_sha": "aaaaaaaa"},
            ]
        },
        "/1/artifacts": {
            "artifacts": [
                {
                    "name": "perf-smoke-report-py3.12",
                    "archive_download_url": "zip1",
                    "expired": False,
                },
                {"name": "other", "archive_download_url": "nope", "expired": False},
            ]
        },
        "/2/artifacts": {
            "artifacts": [
                {
                    "name": "perf-smoke-report-py3.12",
                    "archive_download_url": "zip2",
                    "expired": False,
                },
            ]
        },
        "zip1": artifact_zip(_report(1.0)),
        "zip2": artifact_zip(_report(2.0)),
    }
    seen = []

    def request(url, token, raw=False):
        seen.append((url, token))
        for key, value in runs.items():
            if url.endswith(key) or (key == "runs" and "/workflows/" in url):
                return value
        raise AssertionError(url)

    loaded = perf_history.load_github("o/r", token="t", runs=2, request=request)
    assert [label.split()[1] for label, _ in loaded] == ["aaaaaaa", "bbbbbbb"]
    assert loaded[-1][1]["core_compute/profiles/10x10/loop.leaky"] == pytest.approx(4e-6)
    assert all(token == "t" for _, token in seen)
    assert "status=success" in seen[0][0] and "branch=" not in seen[0][0]


def test_current_reports_join_as_the_newest_run(tmp_path, capsys):
    history = tmp_path / "history"
    history.mkdir()
    for index in range(3):
        (history / f"{index}.json").write_text(json.dumps(_report(1.0, timestamp=index)))
    current = tmp_path / "current.json"
    current.write_text(json.dumps(_report(4.0)))
    assert perf_history.main(["--from-dir", str(history), "--current", str(current)]) == 1
    assert "latest: current" in capsys.readouterr().out
