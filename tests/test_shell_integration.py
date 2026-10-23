import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from lmbash.cli import build_shell_init
from lmbash.config import save_config
from lmbash.providers import ProviderConfig


SHELLS = [shell for shell in ("bash", "zsh") if shutil.which(shell)]


@unittest.skipUnless(SHELLS, "Bash or Zsh is required")
class ShellIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.config_path = self.root / "config" / "lmbash" / "config.json"
        launcher = self.root / "lmbash"
        repo_root = Path(__file__).resolve().parents[1]
        launcher.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            f"sys.path.insert(0, {str(repo_root)!r})\n"
            "from lmbash import cli, config, providers\n"
            "config.getpass.getpass = lambda *args: ''\n"
            "def post_json(url, payload, headers, proxy_url=''):\n"
            "    text = (os.environ['REVIEW_COMMAND'] if len(payload['messages']) > 1\n"
            "            else 'low\\nRead only')\n"
            "    return {'choices': [{'message': {'content': text}}]}\n"
            "providers.post_json = post_json\n"
            "raise SystemExit(cli.main())\n",
            encoding="utf-8",
        )
        launcher.chmod(0o700)
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("LMBASH_")
        }
        self.env.update(
            PATH=str(self.root) + os.pathsep + os.environ["PATH"],
            XDG_CONFIG_HOME=str(self.root / "config"),
            NO_COLOR="1",
        )

    def save_config(self):
        save_config(
            ProviderConfig("openai-compatible", "http://unused.test/v1", "", "test-model"),
            path=self.config_path,
        )

    def invoke(self, shell, args, answers="", command="printf 'confirmed\\n'"):
        return subprocess.run(
            [
                shell, "-f" if shell == "zsh" else "--noprofile", "-c",
                build_shell_init(shell) + '\nlmbash "$@"', "test-shell", *args,
            ],
            input=answers, text=True, capture_output=True, cwd=self.root,
            env=dict(self.env, REVIEW_COMMAND=command), timeout=10,
        )

    def test_first_setup_cancel_does_not_execute_setup_output(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                self.config_path.unlink(missing_ok=True)
                result = self.invoke(shell, ["list", "files"], "1\n4\ntest-model\nn\nn\n")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("Provider type:", result.stderr)
                self.assertIn("Cancelled.", result.stderr)
                self.assertFalse((self.root / "Model:").exists())

    def test_first_setup_executes_only_the_confirmed_command(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                self.config_path.unlink(missing_ok=True)
                result = self.invoke(shell, ["list", "files"], "1\n4\ntest-model\nn\ny\n")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "confirmed\n")
                self.assertIn("Provider type:", result.stderr)

    def test_config_show_displays_saved_config(self):
        self.save_config()
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, ["config", "--show"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("model: test-model", result.stdout)
                self.assertNotIn("Generated command", result.stderr)

    def test_config_reset_removes_saved_config(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                self.save_config()
                result = self.invoke(shell, ["config", "--reset"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(self.config_path.exists())

    def test_config_runs_the_setup_wizard(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, ["config"], "1\n4\nnew-model\nn\n")
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.assertEqual(data["model"], "new-model")
                self.assertIn("Config saved.", result.stdout)

    def test_help_displays_usage_without_executing_it(self):
        for shell in SHELLS:
            for args in (["--help"], ["-h"], ["--model", "test", "--help"]):
                with self.subTest(shell=shell, args=args):
                    result = self.invoke(shell, args)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
                    self.assertEqual(result.stderr, "")

    def test_abbreviated_help_displays_usage_without_executing_it(self):
        for shell in SHELLS:
            for option in ("--h", "--he", "--hel"):
                for args in ([option], ["--model", "test", option], ["show", option]):
                    with self.subTest(shell=shell, args=args):
                        result = self.invoke(shell, args)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage:", result.stdout)
                        self.assertEqual(result.stderr, "")

    def test_short_help_with_suffix_displays_usage_without_executing_it(self):
        for shell in SHELLS:
            for option in ("-help", "-hh", "-hhh", "-hX"):
                for args in ([option], ["--model", "test", option], ["show", option]):
                    with self.subTest(shell=shell, args=args):
                        result = self.invoke(shell, args)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage:", result.stdout)
                        self.assertEqual(result.stderr, "")

    def test_help_after_option_separator_remains_a_generation_request(self):
        self.save_config()
        for shell in SHELLS:
            for option in ("--help", "--he", "-help", "-hh", "-hX"):
                with self.subTest(shell=shell, option=option):
                    result = self.invoke(shell, ["--", option], "y\n")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "confirmed\n")
                    self.assertIn("Generated command", result.stderr)

    def test_exported_arg_is_preserved_in_confirmed_command(self):
        self.save_config()
        self.env["arg"] = "original-value"
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(
                    shell, ["show", "environment"], "y\n", 'printf "<%s>\\n" "$arg"'
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "<original-value>\n")

    def test_shell_init_prints_a_script_without_running_a_generation(self):
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, ["shell-init", shell])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("lmbash()", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_confirmed_commands_use_bash_semantics(self):
        self.save_config()
        commands = [
            ('files="one two"; for file in $files; do printf "<%s>\\n" "$file"; done', "<one>\n<two>\n"),
            ('files=(one two); printf "<%s>\\n" "${files[0]}"', "<one>\n"),
        ]
        for shell in SHELLS:
            for command, expected in commands:
                with self.subTest(shell=shell, command=command):
                    result = self.invoke(shell, ["show", "files"], "y\n", command)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    def test_command_exit_status_is_preserved(self):
        self.save_config()
        for shell in SHELLS:
            with self.subTest(shell=shell):
                result = self.invoke(shell, ["fail"], "y\n", "(exit 7)")
                self.assertEqual(result.returncode, 7, result.stderr)


if __name__ == "__main__":
    unittest.main()
