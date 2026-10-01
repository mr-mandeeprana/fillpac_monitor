"""
Configuration loader for FillPac Operator Monitor.

Loads:
    config/config.yaml

Expands:
    ${VARIABLE}

Environment variables are loaded from:
    1. Existing Windows/process environment
    2. .env beside the executable
    3. .env in the application/project directory
    4. .env beside the bundled application files

The loader is designed to work both:
    - from normal Python development
    - from a PyInstaller packaged EXE

Important:
    Secrets such as CAMERA_PASS and DB_PASSWORD are never logged here.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ----------------------------------------------------------------------
# Environment variable pattern
# ----------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


# ----------------------------------------------------------------------
# Application paths
# ----------------------------------------------------------------------

def _get_executable_dir() -> Path:
    """
    Return the directory containing the running executable.

    For a normal Python run:
        C:\\Digitalization\\fillpac_monitor

    For a PyInstaller EXE:
        C:\\...\\FillPac_Operator_Monitor_v1.0
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


def _get_bundle_dir() -> Path:
    """
    Return PyInstaller's internal bundle directory when available.

    In a PyInstaller application this is usually:

        <EXE_DIR>\\_internal

    In normal Python execution it falls back to the project root.
    """

    meipass = getattr(sys, "_MEIPASS", None)

    if meipass:
        return Path(meipass).resolve()

    return _get_executable_dir()


def _candidate_project_roots() -> list[Path]:
    """
    Return possible application roots in priority order.

    Priority:

        1. EXE directory
        2. PyInstaller bundle directory
        3. Normal source/project directory
    """

    candidates: list[Path] = []

    executable_dir = _get_executable_dir()
    bundle_dir = _get_bundle_dir()

    candidates.append(executable_dir)

    if bundle_dir not in candidates:
        candidates.append(bundle_dir)

    source_root = Path(__file__).resolve().parent.parent

    if source_root not in candidates:
        candidates.append(source_root)

    # Remove duplicates while preserving order.
    unique: list[Path] = []

    for path in candidates:
        path = path.resolve()

        if path not in unique:
            unique.append(path)

    return unique


# ----------------------------------------------------------------------
# Environment loading
# ----------------------------------------------------------------------

def _load_environment(env_path: str | Path | None = None) -> Path | None:
    """
    Load environment variables from .env.

    Existing process environment variables are preserved because
    override=False is used.

    Returns:
        Path of the .env file that was loaded, or None.
    """

    # --------------------------------------------------------------
    # Explicit env file supplied by caller
    # --------------------------------------------------------------

    if env_path is not None:

        explicit_path = Path(env_path).expanduser().resolve()

        if explicit_path.exists():

            load_dotenv(
                dotenv_path=explicit_path,
                override=False,
            )

            return explicit_path

    # --------------------------------------------------------------
    # Search standard application locations
    # --------------------------------------------------------------

    for root in _candidate_project_roots():

        env_file = root / ".env"

        if env_file.exists():

            load_dotenv(
                dotenv_path=env_file,
                override=False,
            )

            return env_file

    # --------------------------------------------------------------
    # Last fallback:
    # python-dotenv searches the current working directory / parents.
    # --------------------------------------------------------------

    load_dotenv(
        override=False,
    )

    return None


# ----------------------------------------------------------------------
# Environment expansion
# ----------------------------------------------------------------------

def _expand_env_vars(value: Any) -> Any:
    """
    Recursively replace ${VAR} in configuration values.

    Example:

        rtsp://${CAMERA_USER}:${CAMERA_PASS}@172.20.45.131:554/...

    becomes:

        rtsp://username:password@172.20.45.131:554/...
    """

    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:

            var_name = match.group(1)

            # Keep the original placeholder if the variable does not exist.
            return os.environ.get(
                var_name,
                match.group(0),
            )

        return _ENV_VAR_PATTERN.sub(
            _sub,
            value,
        )

    if isinstance(value, dict):

        return {
            key: _expand_env_vars(val)
            for key, val in value.items()
        }

    if isinstance(value, list):

        return [
            _expand_env_vars(item)
            for item in value
        ]

    return value


# ----------------------------------------------------------------------
# Configuration validation
# ----------------------------------------------------------------------

def _validate_required_environment_variables(raw: dict) -> None:
    """
    Validate important environment variables used by the application.

    This intentionally does not print secret values.
    """

    camera_source = None

    try:
        camera_source = raw.get("camera", {}).get("source")
    except Exception:
        pass

    if isinstance(camera_source, str):

        unresolved = re.findall(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            camera_source,
        )

        if unresolved:

            missing = []

            for variable in unresolved:

                if not os.environ.get(variable):
                    missing.append(variable)

            if missing:

                raise ConfigError(
                    "Camera configuration contains unresolved "
                    f"environment variables: {', '.join(missing)}. "
                    "Check the .env file beside the EXE or set the "
                    "required Windows environment variables."
                )


# ----------------------------------------------------------------------
# Config error
# ----------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised when application configuration cannot be loaded."""


# ----------------------------------------------------------------------
# AppConfig
# ----------------------------------------------------------------------

@dataclass
class AppConfig:

    raw: dict = field(
        default_factory=dict
    )

    project_root: Path = field(
        default_factory=_get_executable_dir
    )

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(
        self,
        dotted_path: str,
        default: Any = None,
    ) -> Any:
        """
        Fetch a nested configuration value using:

            database.enabled
            camera.source
            models.device
        """

        node: Any = self.raw

        for part in dotted_path.split("."):

            if (
                not isinstance(node, dict)
                or part not in node
            ):
                return default

            node = node[part]

        return node

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def station_id(self) -> str:

        return self.get(
            "station.id",
            "UNKNOWN",
        )

    @property
    def camera_source(self) -> str | None:
        """
        Return the camera source.

        CAMERA_URL takes priority if explicitly defined.
        Otherwise camera.source from config.yaml is used.
        """

        env_override = os.environ.get(
            "CAMERA_URL"
        )

        if env_override:
            return env_override

        return self.get(
            "camera.source"
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def path(
        self,
        relative: str,
    ) -> Path:
        """
        Resolve a relative application path.

        Examples:

            models/yolo26n-pose.pt
            config/config.yaml
            logs/
        """

        p = Path(relative)

        if p.is_absolute():
            return p

        return self.project_root / p

    # ------------------------------------------------------------------
    # Database settings
    # ------------------------------------------------------------------

    def db_settings(self) -> dict:

        return {
            "enabled": self.get(
                "database.enabled",
                True,
            ),

            "driver": self.get(
                "database.driver"
            ),

            "server": os.environ.get(
                "DB_SERVER"
            ),

            "database": os.environ.get(
                "DB_NAME"
            ),

            "user": os.environ.get(
                "DB_USER"
            ),

            "password": os.environ.get(
                "DB_PASSWORD"
            ),

            "trusted_connection": (
                os.environ.get(
                    "DB_TRUSTED_CONNECTION",
                    "false",
                ).lower()
                == "true"
            ),

            "connect_timeout_sec": self.get(
                "database.connect_timeout_sec",
                5,
            ),

            "encrypt": self.get(
                "database.encrypt",
                True,
            ),

            "trust_server_certificate": self.get(
                "database.trust_server_certificate",
                True,
            ),

            "write_queue_max_size": self.get(
                "database.write_queue_max_size",
                5000,
            ),

            "batch_flush_interval_sec": self.get(
                "database.batch_flush_interval_sec",
                2.0,
            ),

            "batch_flush_max_rows": self.get(
                "database.batch_flush_max_rows",
                200,
            ),

            "retry_backoff_sec": self.get(
                "database.retry_backoff_sec",
                [1, 2, 5, 10, 30],
            ),
        }


# ----------------------------------------------------------------------
# Main config loader
# ----------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> AppConfig:
    """
    Load the complete application configuration.

    Development:

        C:\\Digitalization\\fillpac_monitor

    Packaged:

        C:\\...\\FillPac_Operator_Monitor_v1.0

    Expected packaged structure:

        FillPac_Operator_Monitor_v1.0/
        ├── FillPacOperatorMonitor.exe
        ├── .env
        ├── config/
        │   └── config.yaml
        ├── models/
        │   ├── yolo26n-pose.pt
        │   └── hand_landmarker.task
        └── _internal/
    """

    # --------------------------------------------------------------
    # Determine application root
    # --------------------------------------------------------------

    project_root = _get_executable_dir()

    # --------------------------------------------------------------
    # Load environment variables
    # --------------------------------------------------------------

    loaded_env = _load_environment(
        env_path=env_path
    )

    # --------------------------------------------------------------
    # Determine config file
    # --------------------------------------------------------------

    if config_path is not None:

        cfg_file = Path(
            config_path
        ).expanduser()

        if not cfg_file.is_absolute():
            cfg_file = (
                project_root
                / cfg_file
            )

        cfg_file = cfg_file.resolve()

    else:

        possible_configs = []

        for root in _candidate_project_roots():

            possible_configs.append(
                root / "config" / "config.yaml"
            )

        cfg_file = None

        for candidate in possible_configs:

            if candidate.exists():

                cfg_file = candidate.resolve()
                break

        if cfg_file is None:

            searched = "\n".join(
                f"  - {path}"
                for path in possible_configs
            )

            raise ConfigError(
                "Config file not found. Searched:\n"
                f"{searched}"
            )

    # --------------------------------------------------------------
    # Read YAML
    # --------------------------------------------------------------

    try:

        with open(
            cfg_file,
            "r",
            encoding="utf-8",
        ) as fh:

            raw = yaml.safe_load(fh)

    except Exception as exc:

        raise ConfigError(
            f"Could not read config file: "
            f"{cfg_file}: {exc}"
        ) from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):

        raise ConfigError(
            "config.yaml must contain a YAML mapping/object "
            "at the top level."
        )

    # --------------------------------------------------------------
    # Expand ${VARIABLE} placeholders
    # --------------------------------------------------------------

    raw = _expand_env_vars(
        raw
    )

    # --------------------------------------------------------------
    # Validate unresolved variables
    # --------------------------------------------------------------

    _validate_required_environment_variables(
        raw
    )

    # --------------------------------------------------------------
    # Create config object
    # --------------------------------------------------------------

    return AppConfig(
        raw=raw,
        project_root=project_root,
    )