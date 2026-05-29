import os
import tempfile
import unittest
from unittest import mock

import lmbash.cli as lmbash
from lmbash.config import load_config


class CleanupTests(unittest.TestCase):
    def test_clean_command_strips_plain_fence(self):
        self.assertEqual(lmbash.clean_command("```bash\npwd\n```"), "pwd")

    def test_clean_command_strips_generic_fence(self):
        self.assertEqual(lmbash.clean_command("```\nls -la\n```"), "ls -la")

    def test_default_model_uses_gemma_when_env_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(lmbash.default_model(), "google/gemma-4-e4b")

    def test_default_model_uses_env_override(self):
        with mock.patch.dict(os.environ, {"LMSTUDIO_MODEL": "custom/model"}, clear=True):
            self.assertEqual(lmbash.default_model(), "custom/model")


class ApiTests(unittest.TestCase):
    def test_build_payload_requests_one_command(self):
        payload = lmbash.build_payload("show current directory", "google/gemma-4-e4b")
        self.assertEqual(payload["model"], "google/gemma-4-e4b")
        self.assertEqual(payload["temperature"], 0)
        self.assertIn("exactly one bash command", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"], "show current directory")

    @mock.patch("lmbash.cli.urllib.request.urlopen")
    def test_request_command_extracts_assistant_content(self, urlopen):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)
        response.read.return_value = b'{"choices":[{"message":{"content":"pwd"}}]}'
        urlopen.return_value = response

        command = lmbash.request_command(
            "show pwd",
            "http://localhost:1234/v1",
            "google/gemma-4-e4b",
        )

        self.assertEqual(command, "pwd")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:1234/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")

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
    @mock.patch("lmbash.cli.request_command")
    def test_main_refines_command_with_context_before_execute(self, request_command, run_command):
        request_command.side_effect = ["ls", "ls -a"]
        run_command.return_value = mock.Mock(stdout="", stderr="", returncode=0)

        with mock.patch("builtins.input", side_effect=["e", "include hidden files", "y"]):
            exit_code = lmbash.main(["list files"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(request_command.call_count, 2)
        refinement_prompt = request_command.call_args_list[1].args[0]
        self.assertIn("Original request:\nlist files", refinement_prompt)
        self.assertIn("Current command:\nls", refinement_prompt)
        self.assertIn("New requirement:\ninclude hidden files", refinement_prompt)
        run_command.assert_called_once_with("ls -a")

    def test_main_config_saves_interactive_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["1", "4", "", "llama3"]), mock.patch(
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
        ), mock.patch(
            "builtins.input",
            side_effect=["1", "2", "sk-openrouter-secret", "openrouter/model"],
        ), mock.patch("sys.stdout"):
            self.assertEqual(lmbash.main(["config"]), 0)

            with mock.patch("sys.stdout") as stdout:
                exit_code = lmbash.main(["config", "--show"])

        output = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertEqual(exit_code, 0)
        self.assertIn("provider: openai-compatible", output)
        self.assertIn("api_key: sk-o...cret", output)
        self.assertNotIn("sk-openrouter-secret", output)

    def test_main_config_reset_removes_existing_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"XDG_CONFIG_HOME": temp_dir}, clear=True
        ), mock.patch("builtins.input", side_effect=["2", "1", "secret", "claude"]), mock.patch(
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
