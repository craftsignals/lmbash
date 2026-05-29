import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lmbash.config import (
    CLAUDE_COMPATIBLE_PRESETS,
    DEFAULT_CONFIG,
    OPENAI_COMPATIBLE_PRESETS,
    ConfigError,
    apply_env_overrides,
    config_path,
    configure_interactively,
    format_config,
    load_config,
    mask_api_key,
    remove_config,
    save_config,
)
from lmbash.providers import ProviderConfig


class ConfigTests(unittest.TestCase):
    def test_presets_and_default_config_match_supported_providers(self):
        self.assertEqual(
            OPENAI_COMPATIBLE_PRESETS,
            {
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
            },
        )
        self.assertEqual(
            CLAUDE_COMPATIBLE_PRESETS,
            {
                "anthropic": {
                    "base_url": "https://api.anthropic.com",
                    "requires_api_key": True,
                }
            },
        )
        self.assertEqual(
            DEFAULT_CONFIG,
            ProviderConfig(
                provider="openai-compatible",
                base_url="http://localhost:1234/v1",
                api_key="",
                model="local-model",
                preset="lmstudio",
            ),
        )

    def test_save_config_writes_private_json_and_load_config_reads_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "config.json"
            config = ProviderConfig(
                provider="claude-compatible",
                base_url="https://api.anthropic.com",
                api_key="secret",
                model="claude-model",
                preset="anthropic",
            )

            save_config(config, path)
            loaded = load_config(path)

            self.assertEqual(loaded, config)
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)
            with path.open(encoding="utf-8") as file:
                self.assertEqual(
                    json.load(file),
                    {
                        "provider": "claude-compatible",
                        "base_url": "https://api.anthropic.com",
                        "api_key": "secret",
                        "model": "claude-model",
                        "preset": "anthropic",
                    },
                )

    def test_load_config_returns_none_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(load_config(Path(temp_dir) / "missing.json"))

    def test_load_config_rejects_invalid_json_and_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_json = Path(temp_dir) / "invalid.json"
            invalid_json.write_text("{", encoding="utf-8")
            missing_field = Path(temp_dir) / "missing-field.json"
            missing_field.write_text(
                json.dumps(
                    {
                        "provider": "openai-compatible",
                        "base_url": "http://localhost:1234/v1",
                        "api_key": "",
                        "preset": "lmstudio",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(invalid_json)
            with self.assertRaises(ConfigError):
                load_config(missing_field)

    def test_apply_env_overrides_updates_config_without_changing_preset(self):
        config = ProviderConfig(
            provider="openai-compatible",
            base_url="http://localhost:1234/v1",
            api_key="",
            model="local-model",
            preset="lmstudio",
        )
        env = {
            "LMBASH_PROVIDER": "claude-compatible",
            "LMBASH_BASE_URL": "https://api.anthropic.com",
            "LMBASH_API_KEY": "secret",
            "LMBASH_MODEL": "claude-model",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            overridden = apply_env_overrides(config)

        self.assertEqual(
            overridden,
            ProviderConfig(
                provider="claude-compatible",
                base_url="https://api.anthropic.com",
                api_key="secret",
                model="claude-model",
                preset="lmstudio",
            ),
        )
        self.assertEqual(config.preset, "lmstudio")

    def test_mask_api_key_masks_empty_short_and_long_keys(self):
        self.assertEqual(mask_api_key(""), "")
        self.assertEqual(mask_api_key("short"), "****")
        self.assertEqual(mask_api_key("sk-1234567890"), "sk-1...7890")

    def test_remove_config_deletes_existing_file_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            save_config(DEFAULT_CONFIG, path)

            self.assertTrue(remove_config(path))
            self.assertFalse(path.exists())
            self.assertFalse(remove_config(path))

    def test_config_path_uses_xdg_config_home_or_user_config_dir(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/xdg"}, clear=True):
            self.assertEqual(config_path(), Path("/tmp/xdg/lmbash/config.json"))

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "pathlib.Path.home", return_value=Path("/home/example")
        ):
            self.assertEqual(
                config_path(),
                Path("/home/example/.config/lmbash/config.json"),
            )

    def test_configure_interactively_openrouter_preset_requires_api_key(self):
        with mock.patch(
            "builtins.input",
            side_effect=["1", "2", "sk-openrouter-secret", "openrouter/model"],
        ):
            config = configure_interactively()

        self.assertEqual(
            config,
            ProviderConfig(
                provider="openai-compatible",
                preset="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-openrouter-secret",
                model="openrouter/model",
            ),
        )

    def test_configure_interactively_ollama_preset_allows_empty_api_key(self):
        with mock.patch("builtins.input", side_effect=["1", "4", "", "llama3"]):
            config = configure_interactively()

        self.assertEqual(
            config,
            ProviderConfig(
                provider="openai-compatible",
                preset="ollama",
                base_url="http://localhost:11434/v1",
                api_key="",
                model="llama3",
            ),
        )

    def test_configure_interactively_anthropic_preset(self):
        with mock.patch(
            "builtins.input",
            side_effect=["2", "1", "anthropic-secret", "claude-3-5-sonnet"],
        ):
            config = configure_interactively()

        self.assertEqual(
            config,
            ProviderConfig(
                provider="claude-compatible",
                preset="anthropic",
                base_url="https://api.anthropic.com",
                api_key="anthropic-secret",
                model="claude-3-5-sonnet",
            ),
        )

    def test_format_config_masks_api_key(self):
        config = ProviderConfig(
            provider="openai-compatible",
            preset="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-very-secret-key",
            model="gpt-4.1-mini",
        )

        formatted = format_config(config)

        self.assertIn("provider: openai-compatible", formatted)
        self.assertIn("preset: openai", formatted)
        self.assertIn("base_url: https://api.openai.com/v1", formatted)
        self.assertIn("api_key: sk-v...-key", formatted)
        self.assertIn("model: gpt-4.1-mini", formatted)
        self.assertNotIn("sk-very-secret-key", formatted)


if __name__ == "__main__":
    unittest.main()
