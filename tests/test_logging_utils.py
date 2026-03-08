import argparse
import os
from pathlib import Path

from pyRTC.logging_utils import add_logging_cli_args, configure_logging, get_logger


def test_add_logging_cli_args_supports_runtime_overrides():
    parser = argparse.ArgumentParser()
    add_logging_cli_args(parser)

    args = parser.parse_args(["--log-level", "DEBUG", "--log-dir", "logs", "--log-file", "custom.log"])

    assert args.log_level == "DEBUG"
    assert args.log_dir == "logs"
    assert args.log_file == "custom.log"


def test_get_logger_prefixes_non_pyrtc_names():
    logger = get_logger("scripts.view")
    assert logger.name == "pyRTC.scripts.view"


def test_configure_logging_writes_file_and_exports_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PYRTC_LOG_LEVEL", raising=False)
    monkeypatch.delenv("PYRTC_LOG_DIR", raising=False)
    monkeypatch.delenv("PYRTC_LOG_FILE", raising=False)

    logger = configure_logging(
        app_name="pyrtc-test",
        component_name="unit",
        level="INFO",
        log_dir=tmp_path,
        color=False,
        console=False,
    )
    logger.info("hello logging")

    for handler in logger.handlers:
        handler.flush()

    log_files = list(Path(tmp_path).glob("pyrtc-test_unit_*.log"))
    assert len(log_files) == 1
    assert "hello logging" in log_files[0].read_text(encoding="utf-8")
    assert Path(os.environ["PYRTC_LOG_DIR"]).samefile(tmp_path)
    assert os.environ["PYRTC_LOG_LEVEL"] == "INFO"
