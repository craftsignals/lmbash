from dataclasses import asdict
import getpass
import json
import os
from pathlib import Path

from lmbash.providers import ProviderConfig
from lmbash import terminal


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


def _choose_option(prompt, options):
    print(terminal.prompt(prompt))
    for index, label in enumerate(options, start=1):
        print(f"{index}. {label}")

    while True:
        answer = input("> ").strip()
        try:
            choice = int(answer)
        except ValueError:
            print(terminal.status("Invalid selection."))
            continue
        if 1 <= choice <= len(options):
            return options[choice - 1]
        print(terminal.status("Invalid selection."))


def _prompt_required(label):
    value = input(label).strip()
    if not value:
        raise ConfigError(f"{label.rstrip(': ')} cannot be empty")
    return value


def configure_interactively():
    provider_choice = _choose_option(
        "Provider type:",
        ["openai-compatible", "claude-compatible"],
    )

    if provider_choice == "openai-compatible":
        provider = "openai-compatible"
        preset_options = ["openai", "openrouter", "lmstudio", "ollama", "Custom"]
        preset_choice = _choose_option("Preset:", preset_options)
        presets = OPENAI_COMPATIBLE_PRESETS
    else:
        provider = "claude-compatible"
        preset_options = ["anthropic", "Custom"]
        preset_choice = _choose_option("Preset:", preset_options)
        presets = CLAUDE_COMPATIBLE_PRESETS

    if preset_choice == "Custom":
        preset = "custom"
        base_url = _prompt_required("Base URL: ")
        requires_api_key = True
    else:
        preset = preset_choice
        preset_config = presets[preset]
        base_url = preset_config["base_url"]
        requires_api_key = preset_config["requires_api_key"]

    if not base_url:
        raise ConfigError("Base URL cannot be empty")

    api_key = getpass.getpass("API key: ").strip()
    if requires_api_key and not api_key:
        raise ConfigError("API key cannot be empty")

    model = _prompt_required("Model: ")

    return ProviderConfig(
        provider=provider,
        preset=preset,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def format_config(config):
    return "\n".join(
        [
            f"provider: {config.provider}",
            f"preset: {config.preset}",
            f"base_url: {config.base_url}",
            f"api_key: {mask_api_key(config.api_key)}",
            f"model: {config.model}",
        ]
    )
