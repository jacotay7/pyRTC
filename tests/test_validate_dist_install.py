import pyRTC.scripts.validate_dist_install as validate_dist_install


def test_find_built_wheel_returns_latest_sorted_match(tmp_path):
    older = tmp_path / "pyrtcao-0.9.0-py3-none-any.whl"
    newer = tmp_path / "pyrtcao-1.0.0-py3-none-any.whl"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    assert validate_dist_install.find_built_wheel(tmp_path) == newer


def test_build_validation_commands_include_wheel_and_import_check(tmp_path):
    venv_dir = tmp_path / "venv"
    wheel_path = tmp_path / "pyrtcao-1.0.0-py3-none-any.whl"

    commands = validate_dist_install.build_validation_commands(venv_dir, wheel_path)

    assert commands[0][0:3] == [validate_dist_install.sys.executable, "-m", "venv"]
    assert str(wheel_path) in commands[2]
    assert commands[3][1] == "-c"
    assert "import pyRTC" in commands[3][2]
    assert commands[4][-1] == "--help"