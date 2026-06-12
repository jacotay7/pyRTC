"""System orchestration for pyrtc: component runtimes and the RTCManager."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import os
import socket
import threading
import time
from pathlib import Path

from pyrtc.config_runtime import sync_runtime_config
from pyrtc.component_loading import (
    import_symbol as _import_symbol,
    import_symbol_from_file as _import_symbol_from_file,
)
from pyrtc.logging_utils import (
    PYRTC_LOG_DIR_ENV,
    PYRTC_LOG_FILE_ENV,
    add_logging_cli_args,
    configure_logging_from_args,
    get_logger,
)
from pyrtc.rpc import Listener, HardwareLauncher
from pyrtc.streams import reconcile_expected_output_shms
from pyrtc.utils import set_from_config, set_affinity_and_priority

logger = get_logger(__name__)


def work(obj, function_name, affinity):
    """Run one component worker function in a loop while the component lives."""
    set_affinity_and_priority(function_name, [affinity])
    work_function = getattr(obj, function_name, None)
    while obj.alive:
        if obj.running:
            try:
                work_function()
            except Exception:
                component_logger = getattr(obj, "logger", logger)
                component_logger.exception("Worker function '%s' crashed", function_name)
                time.sleep(0.05)
        else:
            time.sleep(1e-3)
    return


def build_component_runtime_config(system_conf: dict, section_name: str) -> dict:
    sync_runtime_config(system_conf)

    conf = dict(system_conf[section_name])
    conf["_sectionName"] = section_name
    conf["_systemStreams"] = dict(system_conf.get("streams", {}))
    conf["_systemConfig"] = system_conf
    return conf


def launch_component(component, conf_key, start=True):
    from pyrtc.config_schema import read_system_config

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")
    add_logging_cli_args(parser)

    # Parse command-line arguments
    args = parser.parse_args()
    configure_logging_from_args(args, app_name=f"pyrtc-{conf_key}", component_name=conf_key)

    system_conf = read_system_config(args.config)
    conf = build_component_runtime_config(system_conf, conf_key)

    set_affinity_and_priority("", set_from_config(conf, "affinity", 0))

    try:
        obj = component(conf=conf)
        if start:
            obj.start()

        listener = Listener(obj, port=int(args.port))
        while listener.running:
            listener.listen()
            time.sleep(1e-3)
    except Exception:
        logger.exception("Failed to launch component %s", conf_key)
        raise


DEFAULT_COMPONENT_ORDER = ("modulator", "wfc", "wfs", "slopes", "loop", "psf", "telemetry")


@dataclass
class ComponentRuntimeStatus:
    section_name: str
    mode: str
    state: str
    component_class: str
    error: str | None = None
    last_error: str | None = None
    port: int | None = None
    target: str | None = None
    pid: int | None = None
    start_time: float | None = None
    uptime_seconds: float = 0.0
    last_heartbeat_time: float | None = None
    last_success_time: float | None = None
    last_failure_time: float | None = None
    restart_count: int = 0
    restart_policy: str = "never"
    log_file: str | None = None
    desired_running: bool = False

    def to_dict(self) -> dict:
        return {
            "section_name": self.section_name,
            "mode": self.mode,
            "state": self.state,
            "component_class": self.component_class,
            "error": self.error,
            "last_error": self.last_error,
            "port": self.port,
            "target": self.target,
            "pid": self.pid,
            "start_time": self.start_time,
            "uptime_seconds": self.uptime_seconds,
            "last_heartbeat_time": self.last_heartbeat_time,
            "last_success_time": self.last_success_time,
            "last_failure_time": self.last_failure_time,
            "restart_count": self.restart_count,
            "restart_policy": self.restart_policy,
            "log_file": self.log_file,
            "desired_running": self.desired_running,
        }


class BaseComponentRuntime:
    def __init__(
        self, section_name, mode, component_class, *, restart_policy="never", heartbeat_timeout=5.0
    ) -> None:
        self.section_name = section_name
        self.mode = mode
        self.component_class = component_class
        self.state = "created"
        self.error = None
        self.last_error = None
        self.pid = None
        self.start_time = None
        self.last_heartbeat_time = None
        self.last_success_time = None
        self.last_failure_time = None
        self.restart_count = 0
        self.restart_policy = restart_policy
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.log_file = None
        self.desired_running = False

    @property
    def component_class_path(self) -> str:
        return f"{self.component_class.__module__}.{self.component_class.__name__}"

    def _set_running(self, *, timestamp: float | None = None) -> None:
        now = time.time() if timestamp is None else float(timestamp)
        self.state = "running"
        self.error = None
        self.start_time = now
        self.last_heartbeat_time = now
        self.last_success_time = now

    def _record_success(self, *, timestamp: float | None = None) -> None:
        now = time.time() if timestamp is None else float(timestamp)
        self.last_heartbeat_time = now
        self.last_success_time = now
        if self.desired_running:
            self.state = "running"
            self.error = None

    def _record_problem(self, message: str, *, state: str) -> None:
        now = time.time()
        self.state = state
        self.error = str(message)
        self.last_error = str(message)
        self.last_failure_time = now

    def _set_stopped(self) -> None:
        self.state = "stopped"
        self.error = None
        self.pid = None
        self.start_time = None

    def _set_built(self) -> None:
        self.state = "built"
        self.error = None
        self.start_time = None

    def build(self) -> None:
        self._set_built()

    def refresh_health(self) -> str:
        return self.state

    def restart(self, *, reason: str | None = None) -> None:
        self.restart_count += 1
        if reason:
            self.last_error = str(reason)
        self.stop()
        self.start()

    def _uptime_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(0.0, time.time() - self.start_time)

    def status(self) -> dict:
        return ComponentRuntimeStatus(
            section_name=self.section_name,
            mode=self.mode,
            state=self.state,
            component_class=self.component_class_path,
            error=self.error,
            last_error=self.last_error,
            pid=self.pid,
            start_time=self.start_time,
            uptime_seconds=self._uptime_seconds(),
            last_heartbeat_time=self.last_heartbeat_time,
            last_success_time=self.last_success_time,
            last_failure_time=self.last_failure_time,
            restart_count=self.restart_count,
            restart_policy=self.restart_policy,
            log_file=self.log_file,
            desired_running=self.desired_running,
        ).to_dict()


class SoftComponentRuntime(BaseComponentRuntime):
    def __init__(
        self,
        section_name,
        component_class,
        conf: dict,
        *,
        shared_resource=None,
        restart_policy="never",
        heartbeat_timeout=5.0,
    ) -> None:
        super().__init__(
            section_name=section_name,
            mode="soft-rtc",
            component_class=component_class,
            restart_policy=restart_policy,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.conf = conf
        self.shared_resource = shared_resource
        self.component = None
        self.pid = os.getpid()
        self.log_file = _resolve_soft_runtime_log_file(component_class)

    def start(self) -> None:
        if self.state == "running":
            return
        self.desired_running = True
        try:
            self.build()
            self.component.start()
            self.pid = os.getpid()
            self._set_running()
        except Exception as exc:
            self._record_problem(str(exc), state="failed")
            raise

    def build(self) -> None:
        if self.component is None:
            if self.shared_resource is None:
                self.component = self.component_class(self.conf)
            else:
                self.component = self.component_class(self.conf, self.shared_resource)
        self.pid = os.getpid()
        self._set_built()

    def stop(self) -> None:
        self.desired_running = False
        if self.component is None:
            self._set_built()
            return
        try:
            self.component.stop()
            self._set_built()
        except Exception as exc:
            self._record_problem(str(exc), state="failed")
            raise

    def refresh_health(self) -> str:
        if self.component is None:
            if self.desired_running:
                self._record_problem("component not initialized", state="failed")
            return self.state

        alive = bool(getattr(self.component, "alive", True))
        running = bool(getattr(self.component, "running", False))
        if self.desired_running and not alive:
            self._record_problem("component reported alive=False", state="failed")
            return self.state
        if self.desired_running and not running:
            self._record_problem("component reported running=False", state="degraded")
            return self.state
        if running:
            self._record_success()
        return self.state


class HardComponentRuntime(BaseComponentRuntime):
    def __init__(
        self,
        section_name,
        component_class,
        script_path: str,
        config_path: str,
        port: int,
        *,
        restart_policy="never",
        heartbeat_timeout=5.0,
        rpc_timeout=0.25,
        log_dir: str | None = None,
        log_file: str | None = None,
        launcher_cls=HardwareLauncher,
    ) -> None:
        super().__init__(
            section_name=section_name,
            mode="hard-rtc",
            component_class=component_class,
            restart_policy=restart_policy,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.script_path = script_path
        self.config_path = config_path
        self.port = port
        self.rpc_timeout = float(rpc_timeout)
        self.log_dir = log_dir
        self.configured_log_file = log_file
        self.launcher_cls = launcher_cls
        self.launcher = None

    def start(self) -> None:
        if self.state == "running":
            return
        self.desired_running = True
        try:
            self.build()
            if self.launcher is None or not _launcher_process_alive(self.launcher):
                self.launcher = self.launcher_cls(
                    self.script_path, self.config_path, self.port, timeout=self.rpc_timeout
                )
                _apply_runtime_logging_environment(
                    log_dir=self.log_dir, log_file=self.configured_log_file
                )
                self.launcher.launch()
            self.launcher.run("start")
            self.pid = _launcher_pid(self.launcher)
            self.log_file = _resolve_hard_runtime_log_file(
                self.section_name,
                pid=self.pid,
                log_dir=self.log_dir,
                log_file=self.configured_log_file,
            )
            self._set_running()
        except Exception as exc:
            self._record_problem(str(exc), state="failed")
            raise

    def build(self) -> None:
        self._set_built()

    def stop(self) -> None:
        self.desired_running = False
        if self.launcher is None:
            self._set_built()
            return
        launcher = self.launcher
        try:
            launcher.run("stop")
        except Exception:
            pass
        try:
            launcher.shutdown()
        except Exception as exc:
            self._record_problem(str(exc), state="failed")
            raise
        self.launcher = None
        self._set_built()

    def refresh_health(self) -> str:
        if not self.desired_running:
            if self.state != "stopped":
                self._set_stopped()
            return self.state

        if self.launcher is None:
            if self.desired_running:
                self._record_problem("launcher not initialized", state="failed")
            return self.state

        self.pid = _launcher_pid(self.launcher)
        if hasattr(self.launcher, "health_check"):
            health = self.launcher.health_check(timeout=self.rpc_timeout)
            last_contact = health.get("last_contact_time")
            state = health.get("state", self.state)
            if state == "running":
                self._record_success(timestamp=last_contact)
            else:
                self._record_problem(health.get("error", "health check failed"), state=state)
            return self.state

        if hasattr(self.launcher, "get_property"):
            try:
                running = self.launcher.get_property("running")
            except Exception as exc:
                self._record_problem(f"health check RPC failed: {exc}", state="degraded")
                return self.state
            if running == -1:
                self._record_problem("health check RPC failed", state="degraded")
                return self.state
            if not bool(running):
                self._record_problem("component reported running=False", state="degraded")
                return self.state

        if self.desired_running:
            self._record_success()
        return self.state

    def status(self) -> dict:
        payload = super().status()
        payload.update({"port": self.port, "target": self.script_path})
        return payload

    def restart(self, *, reason: str | None = None) -> None:
        self.restart_count += 1
        if reason:
            self.last_error = str(reason)
        launcher = self.launcher
        self.launcher = None
        if launcher is not None and hasattr(launcher, "close"):
            try:
                launcher.close(force=True)
            except Exception:
                logger.debug(
                    "Failed to close dead launcher for %s", self.section_name, exc_info=True
                )
        self.start()


def _resolve_soft_runtime_log_file(component_class) -> str | None:
    explicit_log_file = os.environ.get(PYRTC_LOG_FILE_ENV)
    if explicit_log_file:
        return explicit_log_file
    log_dir = os.environ.get(PYRTC_LOG_DIR_ENV)
    if not log_dir:
        return None
    return str(
        (
            Path(log_dir).expanduser() / f"pyrtc_{component_class.__name__}_{os.getpid()}.log"
        ).resolve()
    )


def _resolve_hard_runtime_log_file(
    section_name: str, *, pid: int | None, log_dir: str | None, log_file: str | None
) -> str | None:
    if log_file:
        return str(Path(log_file).expanduser().resolve())
    if pid is None or not log_dir:
        return None
    filename = f"pyrtc-{section_name}_{section_name}_{pid}.log"
    return str((Path(log_dir).expanduser() / filename).resolve())


def _apply_runtime_logging_environment(*, log_dir: str | None, log_file: str | None) -> None:
    if log_file:
        os.environ[PYRTC_LOG_FILE_ENV] = str(Path(log_file).expanduser())
        os.environ.pop(PYRTC_LOG_DIR_ENV, None)
        return
    if log_dir:
        os.environ[PYRTC_LOG_DIR_ENV] = str(Path(log_dir).expanduser())
        os.environ.pop(PYRTC_LOG_FILE_ENV, None)


def _launcher_pid(launcher) -> int | None:
    pid = getattr(launcher, "pid", None)
    if isinstance(pid, int):
        return pid
    process = getattr(launcher, "process", None)
    return getattr(process, "pid", None)


def _launcher_process_alive(launcher) -> bool:
    if hasattr(launcher, "is_process_alive"):
        return bool(launcher.is_process_alive())
    process = getattr(launcher, "process", None)
    if process is None or not hasattr(process, "poll"):
        return True
    return process.poll() is None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _normalize_manager_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    mode_aliases = {
        "soft": "soft-rtc",
        "soft-rtc": "soft-rtc",
        "hard": "hard-rtc",
        "hard-rtc": "hard-rtc",
    }
    if normalized not in mode_aliases:
        raise ValueError("mode must be one of: soft, soft-rtc, hard, hard-rtc")
    return mode_aliases[normalized]


class RTCManager:
    """Validate, launch, stop, and inspect a pyrtc system as one unit.

    The implementation intentionally lives in ``pyrtc.pipeline`` because this
    orchestration layer is an extension of the existing shared-memory and
    launcher runtime rather than a separate subsystem.
    """

    def __init__(
        self,
        config: dict,
        *,
        config_path: str | None = None,
        mode: str | None = None,
        launcher_cls=HardwareLauncher,
    ) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.mode = _normalize_manager_mode(mode)
        if self.mode is not None:
            manager_conf = dict(self.config.get("manager", {}))
            manager_conf["mode"] = self.mode
            self.config["manager"] = manager_conf
        self.launcher_cls = launcher_cls
        self.validated = False
        self.state = "created"
        self.error = None
        self.runtimes = {}
        self.resources = {}
        self._lock = threading.RLock()
        self._supervisor_thread = None
        self._supervisor_stop_event = threading.Event()

    @classmethod
    def from_config_file(
        cls, config_path: str | Path, *, mode: str | None = None, launcher_cls=HardwareLauncher
    ):
        from pyrtc.config_schema import read_system_config

        normalized = read_system_config(config_path)
        manager = cls(
            normalized, config_path=str(config_path), mode=mode, launcher_cls=launcher_cls
        )
        manager.validated = True
        manager.state = "validated"
        return manager

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        config_path: str | None = None,
        mode: str | None = None,
        launcher_cls=HardwareLauncher,
    ):
        return cls(config, config_path=config_path, mode=mode, launcher_cls=launcher_cls)

    def validate(self) -> dict:
        from pyrtc.config_schema import validate_system_config

        self.config = validate_system_config(self.config, config_path=self.config_path)
        self.validated = True
        self.state = "validated"
        self.error = None
        return self.config

    def build(self) -> dict:
        with self._lock:
            if not self.validated:
                self.validate()
            reconcile_expected_output_shms(self.config)
            self._build_runtimes()
            self.state = "building"
            built = []
            try:
                _apply_runtime_logging_environment(
                    log_dir=self._manager_log_dir(),
                    log_file=self._manager_log_file(),
                )
                for section_name in self._component_start_order():
                    runtime = self.runtimes[section_name]
                    resource_section = getattr(runtime, "resource_section", None)
                    if resource_section is not None:
                        provider = self.get_component(resource_section)
                        if provider is None:
                            raise RuntimeError(
                                f"manager: shared resource component '{resource_section}' is not built for '{section_name}'"
                            )
                        runtime.shared_resource = provider
                    runtime.build()
                    built.append(runtime)
            except Exception as exc:
                self.state = "failed"
                self.error = str(exc)
                for runtime in reversed(built):
                    try:
                        runtime.stop()
                    except Exception:
                        pass
                raise
            self.state = "built"
            self.error = None
            return self._status_payload()

    def _component_sections(self) -> list:
        from pyrtc.component_descriptors import list_component_sections

        sections = [section for section in DEFAULT_COMPONENT_ORDER if section in self.config]
        for section in list_component_sections():
            if section in self.config and section not in sections:
                sections.append(section)
        for section_name, section_conf in self.config.items():
            if section_name in {"manager", "streams", "metadata", "resources"}:
                continue
            if isinstance(section_conf, dict) and section_name not in sections:
                sections.append(section_name)
        manager_conf = self.config.get("manager", {})
        for mapping_name in ("component_modes", "ports", "component_classes", "component_files"):
            mapping = manager_conf.get(mapping_name, {})
            if not isinstance(mapping, dict):
                continue
            for section in mapping:
                if section in self.config and section not in sections:
                    sections.append(section)
        return sections

    def _resolve_component_class(self, section_name: str):
        from pyrtc.component_descriptors import get_component_descriptor

        manager_conf = self.config.get("manager", {})
        component_classes = manager_conf.get("component_classes", {})
        component_files = manager_conf.get("component_files", {})
        section_conf = self.config[section_name]
        target = section_conf.get("class_name", component_classes.get(section_name))
        class_file = section_conf.get("class_file", component_files.get(section_name))
        if target is None:
            target = section_conf.get("name")

        if target is not None:
            if inspect.isclass(target):
                return target
            if isinstance(target, str) and "." not in target and class_file:
                component_file = Path(class_file).expanduser()
                if component_file.exists():
                    return _import_symbol_from_file(str(component_file), target)
            return _import_symbol(target)

        descriptor = get_component_descriptor(section_name)
        if descriptor is None:
            raise ValueError(f"No component descriptor found for section '{section_name}'")
        return descriptor.component_class

    def _resolve_script_path(self, section_name: str, component_class) -> str:
        manager_conf = self.config.get("manager", {})
        component_files = manager_conf.get("component_files", {})
        target = component_files.get(section_name)
        if target is not None:
            return str(target)

        source_file = inspect.getsourcefile(component_class)
        if source_file is None:
            raise ValueError(f"Unable to determine script path for section '{section_name}'")
        return source_file

    def _resolve_component_mode(self, section_name: str) -> str:
        manager_conf = self.config.get("manager", {})
        component_modes = manager_conf.get("component_modes", {})
        return component_modes.get(section_name, manager_conf.get("mode", "soft-rtc"))

    def _resolve_resource_class(self, resource_name: str):
        resources_conf = self.config.get("resources", {})
        if not isinstance(resources_conf, dict) or resource_name not in resources_conf:
            raise KeyError(resource_name)
        resource_conf = resources_conf[resource_name]
        target = resource_conf.get("class_name")
        class_file = resource_conf.get("class_file")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"resources.{resource_name}: missing class_name")
        if "." not in target and class_file:
            component_file = Path(class_file).expanduser()
            if component_file.exists():
                return _import_symbol_from_file(str(component_file), target)
        return _import_symbol(target)

    def _build_resources(self) -> None:
        if self.resources:
            return
        resources_conf = self.config.get("resources", {})
        if not isinstance(resources_conf, dict):
            return
        for resource_name, resource_conf in resources_conf.items():
            resource_class = self._resolve_resource_class(resource_name)
            self.resources[resource_name] = resource_class(dict(resource_conf), self.config)

    def _component_resource_name(self, section_name: str) -> str | None:
        section_conf = self.config.get(section_name, {})
        if not isinstance(section_conf, dict):
            return None
        resource_name = section_conf.get("resource")
        if not isinstance(resource_name, str) or not resource_name.strip():
            return None
        return resource_name

    def _component_start_order(self) -> list[str]:
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()
        sections = self._component_sections()
        section_set = set(sections)

        def visit(section_name: str) -> None:
            if section_name in visited:
                return
            if section_name in visiting:
                raise ValueError(f"manager: resource dependency cycle detected at '{section_name}'")
            visiting.add(section_name)
            resource_name = self._component_resource_name(section_name)
            if resource_name in section_set:
                visit(resource_name)
            visiting.remove(section_name)
            visited.add(section_name)
            ordered.append(section_name)

        for section_name in sections:
            visit(section_name)
        return ordered

    def _resolve_port(self, section_name: str) -> int:
        manager_conf = self.config.get("manager", {})
        ports = manager_conf.get("ports", {})
        return int(ports.get(section_name, _find_free_port()))

    def _resolve_restart_policy(self, section_name: str) -> str:
        manager_conf = self.config.get("manager", {})
        component_policies = manager_conf.get("component_restart_policies", {})
        return str(
            component_policies.get(section_name, manager_conf.get("restart_policy", "never"))
        )

    def _resolve_health_check_interval(self) -> float:
        manager_conf = self.config.get("manager", {})
        return float(manager_conf.get("health_check_interval", 1.0))

    def _resolve_heartbeat_timeout(self) -> float:
        manager_conf = self.config.get("manager", {})
        return float(manager_conf.get("heartbeat_timeout", 5.0))

    def _resolve_rpc_timeout(self) -> float:
        manager_conf = self.config.get("manager", {})
        return float(manager_conf.get("rpc_timeout", 0.25))

    def _manager_log_dir(self) -> str | None:
        value = self.config.get("manager", {}).get("log_dir")
        return None if value is None else str(value)

    def _manager_log_file(self) -> str | None:
        value = self.config.get("manager", {}).get("log_file")
        return None if value is None else str(value)

    def _build_runtimes(self) -> None:
        if self.runtimes:
            return
        self._build_resources()
        heartbeat_timeout = self._resolve_heartbeat_timeout()
        rpc_timeout = self._resolve_rpc_timeout()
        manager_log_dir = self._manager_log_dir()
        manager_log_file = self._manager_log_file()
        for section_name in self._component_sections():
            component_class = self._resolve_component_class(section_name)
            mode = self._resolve_component_mode(section_name)
            restart_policy = self._resolve_restart_policy(section_name)
            resource_name = self._component_resource_name(section_name)
            shared_resource = (
                self.resources.get(resource_name) if isinstance(resource_name, str) else None
            )
            if mode == "soft-rtc":
                runtime = SoftComponentRuntime(
                    section_name,
                    component_class,
                    build_component_runtime_config(self.config, section_name),
                    shared_resource=shared_resource,
                    restart_policy=restart_policy,
                    heartbeat_timeout=heartbeat_timeout,
                )
                runtime.resource_section = (
                    resource_name if resource_name in self._component_sections() else None
                )
            else:
                if shared_resource is not None:
                    raise ValueError(
                        f"manager: component '{section_name}' uses shared resource '{resource_name}' and cannot run in hard-rtc mode"
                    )
                if not self.config_path:
                    raise ValueError(
                        f"manager: hard-rtc component '{section_name}' requires a config_path so child processes can load the YAML"
                    )
                runtime = HardComponentRuntime(
                    section_name,
                    component_class,
                    self._resolve_script_path(section_name, component_class),
                    self.config_path,
                    self._resolve_port(section_name),
                    restart_policy=restart_policy,
                    heartbeat_timeout=heartbeat_timeout,
                    rpc_timeout=rpc_timeout,
                    log_dir=manager_log_dir,
                    log_file=manager_log_file,
                    launcher_cls=self.launcher_cls,
                )
            self.runtimes[section_name] = runtime

    def _start_supervisor(self) -> None:
        if self._supervisor_thread is not None and self._supervisor_thread.is_alive():
            return
        self._supervisor_stop_event.clear()
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name="pyrtc-manager-supervisor",
            daemon=True,
        )
        self._supervisor_thread.start()

    def _stop_supervisor(self) -> None:
        self._supervisor_stop_event.set()
        if self._supervisor_thread is None:
            return
        if (
            self._supervisor_thread.is_alive()
            and threading.current_thread() is not self._supervisor_thread
        ):
            self._supervisor_thread.join(
                timeout=max(1.0, self._resolve_health_check_interval() * 2.0)
            )
        self._supervisor_thread = None

    def _supervisor_loop(self) -> None:
        interval = max(self._resolve_health_check_interval(), 0.05)
        while not self._supervisor_stop_event.wait(interval):
            with self._lock:
                if self.state not in {"running", "degraded", "failed"}:
                    continue
                self._refresh_health_locked(supervise=True)

    def _maybe_restart_runtime(self, runtime) -> None:
        if not runtime.desired_running:
            return
        if runtime.restart_policy == "never":
            return
        if runtime.restart_policy == "on-failure" and runtime.state != "failed":
            return
        if runtime.restart_policy == "always" and runtime.state not in {"failed", "stopped"}:
            return
        try:
            runtime.restart(reason=runtime.error)
        except Exception as exc:
            runtime._record_problem(f"restart failed: {exc}", state="failed")

    def _refresh_health_locked(self, *, supervise: bool) -> None:
        if not self.runtimes:
            return

        any_failed = False
        any_degraded = False
        any_running = False
        for section_name in self._component_sections():
            runtime = self.runtimes.get(section_name)
            if runtime is None:
                continue
            runtime.refresh_health()
            if supervise:
                previous_state = runtime.state
                self._maybe_restart_runtime(runtime)
                if previous_state != runtime.state and runtime.state == "running":
                    logger.warning(
                        "Restarted component %s under policy %s",
                        section_name,
                        runtime.restart_policy,
                    )
            if runtime.state == "failed":
                any_failed = True
            elif runtime.state == "degraded":
                any_degraded = True
            elif runtime.state == "running":
                any_running = True

        if any_failed:
            self.state = "failed"
            self.error = (
                "; ".join(
                    f"{section_name}: {runtime.error}"
                    for section_name, runtime in self.runtimes.items()
                    if runtime.state == "failed" and runtime.error
                )
                or self.error
            )
        elif any_degraded:
            self.state = "degraded"
            self.error = (
                "; ".join(
                    f"{section_name}: {runtime.error}"
                    for section_name, runtime in self.runtimes.items()
                    if runtime.state == "degraded" and runtime.error
                )
                or None
            )
        elif any_running:
            self.state = "running"
            self.error = None
        elif any(runtime.state == "built" for runtime in self.runtimes.values()):
            self.state = "built"
            self.error = None
        else:
            self.state = "stopped"
            self.error = None

    def start(self) -> None:
        with self._lock:
            if self.state not in {"built", "running", "degraded", "failed", "starting"}:
                self.build()
            self.state = "starting"
            started = []
            try:
                _apply_runtime_logging_environment(
                    log_dir=self._manager_log_dir(),
                    log_file=self._manager_log_file(),
                )
                for section_name in self._component_start_order():
                    runtime = self.runtimes[section_name]
                    resource_section = getattr(runtime, "resource_section", None)
                    if resource_section is not None:
                        provider = self.get_component(resource_section)
                        if provider is None:
                            raise RuntimeError(
                                f"manager: shared resource component '{resource_section}' is not active for '{section_name}'"
                            )
                        runtime.shared_resource = provider
                    runtime.start()
                    started.append(runtime)
            except Exception as exc:
                self.state = "failed"
                self.error = str(exc)
                for runtime in reversed(started):
                    try:
                        runtime.stop()
                    except Exception:
                        pass
                raise
            self.state = "running"
            self.error = None
            self._start_supervisor()

    def start_component(self, section_name: str) -> None:
        with self._lock:
            if self.state not in {"built", "running", "degraded", "failed", "starting"}:
                self.build()
            runtime = self.runtimes[section_name]
            _apply_runtime_logging_environment(
                log_dir=self._manager_log_dir(),
                log_file=self._manager_log_file(),
            )
            resource_section = getattr(runtime, "resource_section", None)
            if resource_section is not None:
                provider = self.get_component(resource_section)
                if provider is None:
                    raise RuntimeError(
                        f"manager: shared resource component '{resource_section}' is not built for '{section_name}'"
                    )
                runtime.shared_resource = provider
            runtime.start()
            self._refresh_health_locked(supervise=False)
            self._start_supervisor()

    def stop(self) -> None:
        self._stop_supervisor()
        if self.state in {"stopped", "created", "validated"} and not self.runtimes:
            self.state = "stopped"
            return
        with self._lock:
            self.state = "stopping"
            failures = []
            for section_name in reversed(self._component_sections()):
                runtime = self.runtimes.get(section_name)
                if runtime is None:
                    continue
                try:
                    runtime.stop()
                except Exception as exc:
                    failures.append(f"{section_name}: {exc}")
            if failures:
                self.state = "failed"
                self.error = "; ".join(failures)
                raise RuntimeError(self.error)
            self.state = "built" if self.runtimes else "stopped"
            self.error = None

    def stop_component(self, section_name: str) -> None:
        with self._lock:
            runtime = self.runtimes[section_name]
            runtime.stop()
            self._refresh_health_locked(supervise=False)
            if not any(
                component_runtime.desired_running for component_runtime in self.runtimes.values()
            ):
                self._stop_supervisor()

    def restart_component(self, section_name: str) -> None:
        with self._lock:
            runtime = self.runtimes[section_name]
            runtime.restart(reason="manual restart")
            self._refresh_health_locked(supervise=False)

    def _status_payload(self) -> dict:
        return {
            "state": self.state,
            "mode": self.config.get("manager", {}).get("mode", "soft-rtc"),
            "validated": self.validated,
            "config_path": self.config_path,
            "error": self.error,
            "components": {
                section_name: runtime.status() for section_name, runtime in self.runtimes.items()
            },
        }

    def status(self) -> dict:
        with self._lock:
            if self.state in {"running", "degraded", "failed"}:
                self._refresh_health_locked(supervise=False)
            return self._status_payload()

    def refresh_health(self) -> dict:
        with self._lock:
            self._refresh_health_locked(supervise=True)
            return self._status_payload()

    def latency(
        self,
        *,
        source_shm: str | None = None,
        target_shm: str | None = None,
        stream_path: list[str] | tuple[str, ...] | None = None,
        samples: int = 2048,
        show_progress: bool = False,
    ) -> dict:
        from pyrtc.component_descriptors import describe_component_class, get_component_descriptor
        from pyrtc.latency import infer_stream_path, measure_stream_path_latency

        with self._lock:
            if not self.validated:
                self.validate()
            section_names = tuple(self._component_sections())

            def _descriptor_resolver(section_name: str):
                try:
                    descriptor = describe_component_class(
                        self._resolve_component_class(section_name)
                    )
                except Exception:
                    return get_component_descriptor(section_name)
                # A class without a registered descriptor yields a synthesized
                # descriptor with no stream wiring; the section-name descriptor
                # still knows the canonical inputs/outputs in that case.
                if (
                    descriptor is not None
                    and not descriptor.input_streams
                    and not descriptor.output_streams
                ):
                    return get_component_descriptor(section_name) or descriptor
                return descriptor

        if stream_path is not None:
            path = [str(stream_name) for stream_name in stream_path]
            inferred_path = False
        else:
            path, inferred_path = infer_stream_path(
                section_names=section_names,
                descriptor_resolver=_descriptor_resolver,
                source_shm=source_shm,
                target_shm=target_shm,
            )

        report, _ = measure_stream_path_latency(
            path,
            samples=samples,
            show_progress=show_progress,
        )
        payload = report.to_dict()
        payload["inferred_path"] = bool(inferred_path)
        return payload

    def get_component(self, section_name: str):
        runtime = self.runtimes[section_name]
        return getattr(runtime, "component", getattr(runtime, "launcher", None))
