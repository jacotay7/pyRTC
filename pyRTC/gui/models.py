"""Pure-Python GUI data models used by the manager UI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNodeModel:
    section_name: str
    title: str
    subtitle: str
    x: float
    y: float
    state: str = "stopped"
    mode: str = "soft-rtc"
    error: str | None = None
    restart_policy: str | None = None
    input_streams: tuple[str, ...] = ()
    output_streams: tuple[str, ...] = ()
    can_start: bool = False
    can_stop: bool = False


@dataclass(frozen=True)
class GraphEdgeModel:
    source_section: str
    target_section: str
    source_stream: str
    target_stream: str

    @property
    def label(self) -> str:
        if self.source_stream == self.target_stream:
            return self.source_stream
        return f"{self.source_stream} -> {self.target_stream}"


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: tuple[GraphNodeModel, ...] = ()
    edges: tuple[GraphEdgeModel, ...] = ()
    state: str = "created"
    error: str | None = None
    config_path: str | None = None
    mode: str = "soft-rtc"
    metadata: dict[str, object] = field(default_factory=dict)