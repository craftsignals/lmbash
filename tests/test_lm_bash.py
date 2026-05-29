import os
import tempfile
import unittest
from unittest import mock

import lmbash.cli as lmbash
from lmbash.config import load_config
from lmbash.providers import ProviderConfig


class CleanupTests(unittest.TestCase):
    def test_clean_command_strips_plain_fence(self):
        self.assertEqual(lmbash.clean_command("```bash\npwd\n```"), "pwd")

    def test_clean_command_strips_generic_fence(self):
        self.assertEqual(lmbash.clean_command("```\nls -la\n```"), "ls -la")


class ApiTests(unittest.TestCase):
    def test_build_refinement_prompt_includes_context(self):
        prompt = lmbash.build_refinement_prompt(
            "list files",
            "ls",
            "include hidden files",
        )

        self.assertIn("Original request:\nlist files", prompt)
        self.assertIn("Current command:\nls", prompt)
        self.assertIn("New requirement:\ninclude hidden files", prompt)
        self.assertIn("updated bash command", prompt)


class CliTests(unittest.TestCase):
    def test_get_prompt_joins_positional_words(self):
        args = lmbash.parse_args(["list", "files"])
        self.assertEqual(lmbash.prompt_from_args(args), "list files")

    def test_choose_action_accepts_yes(self):
        with mock.patch("builtins.input", return_value="yes"):
            self.assertEqual(lmbash.choose_action(), "execute")

    def test_choose_action_rejects_empty(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(lmbash.choose_action(), "cancel")

    def test_choose_action_accepts_edit(self):
        with mock.patch("builtins.input", return_value="e"):
            self.assertEqual(lmbash.choose_action(), "edit")

    @mock.patch("lmbash.cli.run_command")
    @mock.patch("lmbash.cli.build_client")
    @mock.patch("lmbash.cli.load_effective_config")
    def test_main_refines_command_with_context_before_execute(
        self, load_effective_config, build_client, run_command
    ):
        client = mock.Mock()
        client.request_command.side_effect = ["ls", "ls -a"]
        build_client.return_value = client
        load_effective_config.return_value = ProviderConfig(
            provider="openai-compatible",
            base_url="http://localhost:1234/v1",
            api_key="",
            model="local-model",
            preset="lmstudio",
        )
        run_command.return_value = mock.Mock(stdout="", stderr="", returncode=0)

        with mock.patch("builtins.input", side_effect=["e", "include hidden files", "y"]), mock.patch(
            "sys.stdout"
        ):
            exit_code = lmbash.main(["list files"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(client.request_command.call_count, 2)
        refinement_prompt = client.request_command.call_args_list[1].args[0]
        self.assertIn("Original request:\nlist files", refinement_prompt)
        self.assertIn("Current command:\nls", refinement_prompt)
        self.assertIn("New requirement:\ninclude hidden files", refinement_prompt)
        run_command.assert_called_once_with("ls -a")

    @mock.patch("lmbash.cli.run_command")
    @mock.patch("lmbash.cli.build_client")
    @mock.patch("lmbash.cli.load_effective_config")
    def test_main_uses_client_validated_command_without_recleaning(
        self, load_effective_config, build_client, run_command
    ):
        client = mock.Mock()
        client.request_command.return_value = "```bash\npwd\n```"
        build_client.return_value = client
        load_effective_config.return_value = ProviderConfig(
            provider="openai-compatible",
            base_url="http://localhost:1234/v1",
            api_key="",
            model="local-model",
            preset="lmstudio",
        )
        run_command.return_value = mock.Mock(stdout="", stderr="", returncode=0)

        with mock.patch("builtins.input", return_value="y"), mock.patch("sys.stdout"):
            exit_code = lmbash.main(["show pwd"])

        self.assertEqual(exit_code, 0)
        run_command.assert_called_once_with("```bash\npwd\n```")

    @mock.patch("lmbash.cli.run_command")
    @mock.patch("lmbash.cli.build_client")
    def test_main_missing_config_configures_saves_and_executes_prompt(self, build_client, run_command):
        client = mock.Mock()
        client.request_command.return_value = "pwd"
        build_client.return_value = client
        run_command.return_value = mock.Mock(stdout="", stderr="", returncode=0)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["1", "4", "local-model", "y"]), mock.patch(
            "getpass.getpass", return_value=""
        ), mock.patch(
            "sys.stdout"
        ) as stdout:
            exit_code = lmbash.main(["show pwd"])
            saved_config = load_config()

        output = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertEqual(exit_code, 0)
        self.assertIn("No lmbash config found. Starting setup.", output)
        self.assertEqual(saved_config.provider, "openai-compatible")
        self.assertEqual(saved_config.preset, "ollama")
        self.assertEqual(saved_config.model, "local-model")
        build_client.assert_called_once_with(saved_config)
        client.request_command.assert_called_once_with("show pwd")
        run_command.assert_called_once_with("pwd")

    @mock.patch("lmbash.cli.build_client")
    def test_main_saved_config_applies_cli_base_url_and_model_overrides(self, build_client):
        client = mock.Mock()
        client.request_command.return_value = "pwd"
        build_client.return_value = client

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", return_value="n"), mock.patch("sys.stdout"):
            lmbash.save_config(
                ProviderConfig(
                    provider="claude-compatible",
                    base_url="https://saved.example.test",
                    api_key="secret",
                    model="saved-model",
                    preset="anthropic",
                )
            )

            exit_code = lmbash.main(
                [
                    "--base-url",
                    "https://override.example.test",
                    "--model",
                    "override-model",
                    "show pwd",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = build_client.call_args.args[0]
        self.assertEqual(config.provider, "claude-compatible")
        self.assertEqual(config.base_url, "https://override.example.test")
        self.assertEqual(config.model, "override-model")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.preset, "anthropic")

    def test_main_config_saves_interactive_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["1", "4", "llama3"]), mock.patch(
            "getpass.getpass", return_value=""
        ), mock.patch(
            "sys.stdout"
        ) as stdout:
            exit_code = lmbash.main(["config"])
            saved_config = load_config()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.write.call_args_list[-2].args[0], "Config saved.")
        self.assertEqual(saved_config.provider, "openai-compatible")
        self.assertEqual(saved_config.preset, "ollama")

    def test_main_config_show_prints_masked_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["1", "2", "openrouter/model"]), mock.patch(
            "getpass.getpass", return_value="sk-openrouter-secret"
        ), mock.patch("sys.stdout"):
            self.assertEqual(lmbash.main(["config"]), 0)

            with mock.patch("sys.stdout") as stdout:
                exit_code = lmbash.main(["config", "--show"])

        output = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertEqual(exit_code, 0)
        self.assertIn("provider: openai-compatible", output)
        self.assertIn("api_key: sk-o...cret", output)
        self.assertNotIn("sk-openrouter-secret", output)

    @mock.patch("lmbash.cli.load_config", side_effect=lmbash.ConfigError("bad config"))
    def test_main_config_show_handles_config_error_cleanly(self, load_config):
        with mock.patch("sys.stderr") as stderr:
            exit_code = lmbash.main(["config", "--show"])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Error: bad config",
            "".join(call.args[0] for call in stderr.write.call_args_list),
        )

    @mock.patch("lmbash.cli.save_config", side_effect=lmbash.ConfigError("cannot write config"))
    @mock.patch("lmbash.cli.configure_interactively")
    def test_main_config_save_failure_returns_one(self, configure_interactively, save_config):
        configure_interactively.return_value = mock.Mock()

        with mock.patch("sys.stderr") as stderr:
            exit_code = lmbash.main(["config"])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Error: cannot write config",
            "".join(call.args[0] for call in stderr.write.call_args_list),
        )

    def test_main_config_reset_removes_existing_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["2", "1", "claude"]), mock.patch(
            "getpass.getpass", return_value="secret"
        ), mock.patch(
            "sys.stdout"
        ):
            self.assertEqual(lmbash.main(["config"]), 0)
            self.assertIsNotNone(load_config())

            with mock.patch("sys.stdout") as stdout:
                exit_code = lmbash.main(["config", "--reset"])

            self.assertEqual(exit_code, 0)
            self.assertIsNone(load_config())
            self.assertIn(
                "Config removed.",
                "".join(call.args[0] for call in stdout.write.call_args_list),
            )
