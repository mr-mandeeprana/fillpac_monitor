"""
Loads config/config.yaml, expands ${VAR} placeholders using environment
variables (populated from .env via python-dotenv), and exposes a single
immutable AppConfig object used throughout the app.
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} in strings with os.environ values."""
    if isinstance(value, str):
        def _sub(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


class ConfigError(RuntimeError):
    pass


@dataclass
class AppConfig:
    raw: dict = field(default_factory=dict)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Fetch a nested value using 'a.b.c' notation."""
        node: Any = self.raw
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ---- Convenience accessors for the most commonly used values ----
    @property
    def station_id(self) -> str:
        return self.get("station.id", "UNKNOWN")

    @property
    def camera_source(self) -> str:
        # Environment variable CAMERA_URL always wins if present.
        env_override = os.environ.get("CAMERA_URL")
        if env_override:
            return env_override
        return self.get("camera.source")

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.project_root / p

    def db_settings(self) -> dict:
        return {
            "enabled": self.get("database.enabled", True),
            "driver": self.get("database.driver"),
            "server": os.environ.get("DB_SERVER"),
            "database": os.environ.get("DB_NAME"),
            "user": os.environ.get("DB_USER"),
            "password": os.environ.get("DB_PASSWORD"),
            "trusted_connection": os.environ.get("DB_TRUSTED_CONNECTION", "false").lower() == "true",
            "connect_timeout_sec": self.get("database.connect_timeout_sec", 5),
            "encrypt": self.get("database.encrypt", True),
            "trust_server_certificate": self.get("database.trust_server_certificate", True),
            "write_queue_max_size": self.get("database.write_queue_max_size", 5000),
            "batch_flush_interval_sec": self.get("database.batch_flush_interval_sec", 2.0),
            "batch_flush_max_rows": self.get("database.batch_flush_max_rows", 200),
            "retry_backoff_sec": self.get("database.retry_backoff_sec", [1, 2, 5, 10, 30]),
        }


def load_config(config_path: str | Path | None = None, env_path: str | Path | None = None) -> AppConfig:
    project_root = Path(__file__).resolve().parent.parent

    env_file = Path(env_path) if env_path else project_root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        # Fall back to whatever is already in the process environment
        load_dotenv()

    cfg_file = Path(config_path) if config_path else project_root / "config" / "config.yaml"
    if not cfg_file.exists():
        raise ConfigError(f"Config file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raw = _expand_env_vars(raw)
    return AppConfig(raw=raw, project_root=project_root)
