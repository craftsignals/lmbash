import json
import unittest
from unittest import mock

from lmbash.providers import LmBashError, ProviderConfig, build_client


class ProviderClientTests(unittest.TestCase):
    def make_response(self, body):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=None)
        response.read.return_value = json.dumps(body).encode("utf-8")
        return response

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_posts_chat_completion_request_and_parses_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "pwd"}}]}
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1/",
                api_key="secret-key",
                model="example-model",
            )
        )

        command = client.complete_command("show current directory")

        self.assertEqual(command, "pwd")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(request.headers["Authorization"], "Bearer secret-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "example-model")
        self.assertEqual(body["temperature"], 0)
        self.assertIn("exactly one bash command", body["messages"][0]["content"])
        self.assertEqual(body["messages"][1], {"role": "user", "content": "show current directory"})

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_omits_authorization_header_when_api_key_empty(self, urlopen):
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "pwd"}}]}
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="",
                model="example-model",
            )
        )

        client.complete_command("show current directory")

        request = urlopen.call_args.args[0]
        self.assertNotIn("Authorization", request.headers)

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_cleans_fenced_markdown_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "```bash\npwd\n```"}}]}
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
            )
        )

        command = client.complete_command("show current directory")

        self.assertEqual(command, "pwd")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_rejects_empty_output_after_cleanup(self, urlopen):
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "   "}}]}
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
            )
        )

        with self.assertRaises(LmBashError):
            client.complete_command("show current directory")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_claude_posts_message_request_and_parses_text_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"content": [{"type": "thinking", "text": "ignored"}, {"type": "text", "text": "ls -la"}]}
        )
        client = build_client(
            ProviderConfig(
                provider="claude-compatible",
                base_url="https://claude.example.test/",
                api_key="claude-key",
                model="claude-model",
            )
        )

        command = client.complete_command("list files")

        self.assertEqual(command, "ls -la")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://claude.example.test/v1/messages")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(request.headers["X-api-key"], "claude-key")
        self.assertEqual(request.headers["Anthropic-version"], "2023-06-01")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "claude-model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertIn("exactly one bash command", body["system"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "list files"}])

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_claude_cleans_fenced_markdown_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"content": [{"type": "text", "text": "```\nls -la\n```"}]}
        )
        client = build_client(
            ProviderConfig(
                provider="claude-compatible",
                base_url="https://claude.example.test",
                api_key="claude-key",
                model="claude-model",
            )
        )

        command = client.complete_command("list files")

        self.assertEqual(command, "ls -la")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_claude_rejects_empty_output_after_cleanup(self, urlopen):
        urlopen.return_value = self.make_response(
            {"content": [{"type": "text", "text": "```\n\n```"}]}
        )
        client = build_client(
            ProviderConfig(
                provider="claude-compatible",
                base_url="https://claude.example.test",
                api_key="claude-key",
                model="claude-model",
            )
        )

        with self.assertRaises(LmBashError):
            client.complete_command("list files")

    def test_unknown_provider_raises_lmbash_error(self):
        config = ProviderConfig(
            provider="unknown",
            base_url="https://api.example.test",
            api_key="secret-key",
            model="example-model",
        )

        with self.assertRaises(LmBashError):
            build_client(config)


if __name__ == "__main__":
    unittest.main()
