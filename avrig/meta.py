"""Session metadata (who, which vehicle, where, conditions) and operator presets.

Stored as meta.json in every session and copied into the manifest, so a dataset can be
searched and traced without opening the video. Presets keep entry fast and consistent:
~/.config/avrig/presets.yaml

    operators: [Alice, Bob]
    vehicles: [MG-Windsor-07]
    routes: [city loop, highway loop]
    tags: [night, rain, highway]
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

import yaml

from .errors import ConfigError

DEFAULT_PRESETS = {
    "operators": [], "vehicles": [], "routes": [],
    "weather": ["clear", "cloudy", "rain", "heavy rain", "fog", "haze"],
    "light": ["day", "dusk", "dawn", "night", "night (lit)"],
    "road": ["urban", "highway", "rural", "parking", "test track"],
    "tags": [],
}


@dataclass
class SessionMeta:
    operator: str = ""
    vehicle: str = ""
    route: str = ""
    weather: str = ""
    light: str = ""
    road: str = ""
    notes: str = ""
    tags: list = field(default_factory=list)

    def validate(self):
        for k, v in asdict(self).items():
            if k == "tags":
                if not all(isinstance(t, str) and re.fullmatch(r"[\w .-]{1,40}", t) for t in v):
                    raise ConfigError(f"tags must be short words, got {v}")
            elif not isinstance(v, str) or len(v) > (2000 if k == "notes" else 120):
                raise ConfigError(f"metadata {k!r} must be text ({'2000' if k == 'notes' else '120'} chars max)")
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_pairs(cls, pairs: list[str], base: "SessionMeta | None" = None) -> "SessionMeta":
        """KEY=VALUE strings (CLI --meta); tags accumulate."""
        m = base or cls()
        for p in pairs:
            if "=" not in p:
                raise ConfigError(f"--meta expects KEY=VALUE, got {p!r}")
            k, v = (x.strip() for x in p.split("=", 1))
            if k == "tags" or k == "tag":
                m.tags += [t.strip() for t in v.split(",") if t.strip()]
            elif k in asdict(m):
                setattr(m, k, v)
            else:
                raise ConfigError(f"unknown metadata key {k!r}", f"one of {', '.join(asdict(m))}")
        return m.validate()


def load_presets(path: str) -> dict:
    out = {k: list(v) for k, v in DEFAULT_PRESETS.items()}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"cannot read presets {path}: {e}") from None
        for k, v in data.items():
            if k not in out or not isinstance(v, list):
                raise ConfigError(f"{path}: {k!r} must be one of {list(out)} with a list value")
            out[k] = [str(x) for x in v]
    return out
