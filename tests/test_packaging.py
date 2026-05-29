import tomllib
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_pyproject_declares_lmbash_console_script(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["name"], "lmbash")
        self.assertEqual(data["project"]["requires-python"], ">=3.9")
        self.assertEqual(data["project"]["scripts"]["lmbash"], "lmbash.cli:main")

    def test_readme_documents_multi_provider_configuration(self):
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")

        self.assertIn("openai-compatible", readme_text)
        self.assertIn("claude-compatible", readme_text)
        self.assertIn("lmbash config", readme_text)
        self.assertIn("lmbash config --show", readme_text)
        self.assertIn("lmbash config --reset", readme_text)
        self.assertIn("~/.config/lmbash/config.json", readme_text)
        self.assertIn("0600", readme_text)
        self.assertIn("masks", readme_text)
        self.assertIn("https://api.openai.com/v1", readme_text)
        self.assertIn("https://openrouter.ai/api/v1", readme_text)
        self.assertIn("http://localhost:1234/v1", readme_text)
        self.assertIn("http://localhost:11434/v1", readme_text)
        self.assertIn("https://api.anthropic.com", readme_text)
        self.assertIn("setup wizard", readme_text)
        self.assertIn("LMBASH_PROVIDER", readme_text)
        self.assertIn("LMBASH_BASE_URL", readme_text)
        self.assertIn("LMBASH_API_KEY", readme_text)
        self.assertIn("LMBASH_MODEL", readme_text)
        self.assertIn("Review every command", readme_text)
        self.assertIn("python3 -m unittest -v", readme_text)
        self.assertIn("python3 -m build", readme_text)

    def test_pyproject_description_and_keywords_are_provider_agnostic(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        self.assertEqual(
            data["project"]["description"],
            "Generate and optionally run bash commands using an LLM provider.",
        )
        self.assertNotIn("LM Studio", data["project"]["description"])
        self.assertIn("openrouter", data["project"]["keywords"])
        self.assertIn("ollama", data["project"]["keywords"])

    def test_mit_license_file_exists(self):
        license_path = Path(__file__).resolve().parents[1] / "LICENSE"
        license_text = license_path.read_text(encoding="utf-8")

        self.assertIn("MIT License", license_text)
        self.assertIn("Permission is hereby granted", license_text)


if __name__ == "__main__":
    unittest.main()
