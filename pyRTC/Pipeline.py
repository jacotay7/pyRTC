"""Aggregation shim for pyRTC's runtime infrastructure.

The historical ``pyRTC.Pipeline`` module grew to hold several unrelated
concerns. The implementation now lives in focused modules:

- :mod:`pyRTC.streams` — pyshmem-backed stream policy and SHM planning
- :mod:`pyRTC.rpc` — hard-RTC launcher/listener JSON socket protocol
- :mod:`pyRTC.manager` — component runtimes and :class:`RTCManager`

This module re-exports the public names so existing imports keep working.
Prefer importing from the focused modules in new code.
"""

from pyRTC.component_loading import (  # noqa: F401
    import_symbol as _import_symbol,
    import_symbol_from_file as _import_symbol_from_file,
)
from pyRTC.manager import (  # noqa: F401
    DEFAULT_COMPONENT_ORDER,
    BaseComponentRuntime,
    ComponentRuntimeStatus,
    HardComponentRuntime,
    RTCManager,
    SoftComponentRuntime,
    build_component_runtime_config,
    launchComponent,
    work,
)
from pyRTC.rpc import (  # noqa: F401
    Listener,
    _socket_read_json,
    _socket_send_json,
    hardwareLauncher,
)
from pyRTC.streams import (  # noqa: F401
    TORCH_AVAILABLE,
    _existing_shm_spec,
    _stream_aliases,
    clear_shms,
    create_stream,
    expected_output_shm_specs_for_config,
    expected_output_shms_for_config,
    gpu_torch_available,
    normalize_gpu_device,
    open_stream,
    reconcile_expected_output_shms,
)
