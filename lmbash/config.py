from dataclasses import asdict
import json
import os
from pathlib import Path

from lmbash.providers import ProviderConfig


OPENAI_COMPATIBLE_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "requires_api_key": True,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "requires_api_key": False,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "requires_api_key": False,
    },
}

CLAUDE_COMPATIBLE_PRESETS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "requires_api_key": True,
    },
}

DEFAULT_CONFIG = ProviderConfig(
    provider="openai-compatible",
    base_url="http://localhost:1234/v1",
    api_key="",
    model="local-model",
    preset="lmstudio",
)


class ConfigError(Exception):
    pass


def config_path():
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "lmbash" / "config.json"
    return Path.home() / ".config" / "lmbash" / "config.json"


def load_config(path=None):
    path = Path(path) if path is not None else config_path()
    if not path.exists():
        return None

    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid config JSON: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config must be a JSON object")

    required_fields = ("provider", "base_url", "api_key", "model")
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ConfigError(f"Config missing required field: {missing_fields[0]}")

    values = {field: data[field] for field in required_fields}
    values["preset"] = data.get("preset", "custom")
    for field, value in values.items():
        if not isinstance(value, str):
            raise ConfigError(f"Config field must be a string: {field}")

    return ProviderConfig(**values)


def save_config(config, path=None):
    path = Path(path) if path is not None else config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)
        file.write("\n")
    path.chmod(0o600)


def remove_config(path=None):
    path = Path(path) if path is not None else config_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def apply_env_overrides(config):
    return ProviderConfig(
        provider=os.environ.get("LMBASH_PROVIDER", config.provider),
        base_url=os.environ.get("LMBASH_BASE_URL", config.base_url),
        api_key=os.environ.get("LMBASH_API_KEY", config.api_key),
        model=os.environ.get("LMBASH_MODEL", config.model),
        preset=config.preset,
    )


def mask_api_key(api_key):
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"
