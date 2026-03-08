import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger


logger = get_logger(__name__)


def find_built_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("pyrtcao-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No built wheel matching 'pyrtcao-*.whl' found in {dist_dir}")
    return wheels[-1]


def python_in_venv(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def build_validation_commands(venv_dir: Path, wheel_path: Path) -> list[list[str]]:
    python_exe = str(python_in_venv(venv_dir))
    return [
        [sys.executable, "-m", "venv", str(venv_dir)],
        [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
        [python_exe, "-m", "pip", "install", str(wheel_path)],
        [
            python_exe,
            "-c",
            (
                "import importlib.metadata as md; "
                "import pyRTC; "
                "print('built wheel import OK'); "
                "print(md.version('pyrtcao')); "
                "print(sorted(pyRTC.__all__)[:5])"
            ),
        ],
        [python_exe, "-m", "benchmarks.core_compute_bench", "--help"],
    ]


def run_command(command: list[str]) -> None:
    logger.info("Running command: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            logger.error("Command stdout:\n%s", completed.stdout)
        if completed.stderr:
            logger.error("Command stderr:\n%s", completed.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    if completed.stdout:
        logger.info("Command output:\n%s", completed.stdout.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate that the built pyrtcao wheel installs and imports in a clean venv.")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing built distribution artifacts")
    parser.add_argument("--venv-dir", default=None, help="Optional explicit virtualenv path to create and keep")
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-validate-dist", component_name="validate_dist_install")

    dist_dir = Path(args.dist_dir).expanduser().resolve()
    wheel_path = find_built_wheel(dist_dir)
    logger.info("Using built wheel %s", wheel_path)

    if args.venv_dir:
        venv_dir = Path(args.venv_dir).expanduser().resolve()
        if venv_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing virtualenv path: {venv_dir}")
        commands = build_validation_commands(venv_dir, wheel_path)
        for command in commands:
            run_command(command)
        logger.info("Validated built wheel using persistent virtualenv %s", venv_dir)
        return 0

    with tempfile.TemporaryDirectory(prefix="pyrtc-wheel-test-") as tmp_dir:
        venv_dir = Path(tmp_dir) / "venv"
        commands = build_validation_commands(venv_dir, wheel_path)
        for command in commands:
            run_command(command)
        logger.info("Validated built wheel using temporary virtualenv %s", venv_dir)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())