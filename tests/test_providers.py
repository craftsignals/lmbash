import json
import socket
import sys
import unittest
import urllib.error
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

        command = client.request_command("show current directory")

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

        client.request_command("show current directory")

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

        command = client.request_command("show current directory")

        self.assertEqual(command, "pwd")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_uses_last_non_empty_line_from_multi_command_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "ls\nls -a"}}]}
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
            )
        )

        command = client.request_command("list hidden files")

        self.assertEqual(command, "ls -a")

    @mock.patch("lmbash.providers.urllib.request.build_opener")
    def test_openai_uses_http_proxy_handler_when_proxy_url_is_http(self, build_opener):
        opener = mock.Mock()
        opener.open.return_value = self.make_response(
            {"choices": [{"message": {"content": "pwd"}}]}
        )
        build_opener.return_value = opener
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
                proxy_url="http://127.0.0.1:7890",
            )
        )

        command = client.request_command("show pwd")

        self.assertEqual(command, "pwd")
        handler = build_opener.call_args.args[0]
        self.assertEqual(handler.proxies["http"], "http://127.0.0.1:7890")
        self.assertEqual(handler.proxies["https"], "http://127.0.0.1:7890")
        opener.open.assert_called_once()

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_openai_uses_socks_proxy_socket_temporarily(self, urlopen):
        class FakeSocks:
            SOCKS5 = object()
            socksocket = object()

            def __init__(self):
                self.calls = []

            def set_default_proxy(self, proxy_type, host, port, rdns):
                self.calls.append((proxy_type, host, port, rdns))

        fake_socks = FakeSocks()
        urlopen.return_value = self.make_response(
            {"choices": [{"message": {"content": "pwd"}}]}
        )
        original_socket = socket.socket
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
                proxy_url="socks5h://127.0.0.1:7890",
            )
        )

        with mock.patch.dict(sys.modules, {"socks": fake_socks}):
            command = client.request_command("show pwd")

        self.assertEqual(command, "pwd")
        self.assertEqual(fake_socks.calls, [(fake_socks.SOCKS5, "127.0.0.1", 7890, True)])
        self.assertIs(socket.socket, original_socket)

    def test_socks_proxy_without_pysocks_raises_lmbash_error(self):
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
                model="example-model",
                proxy_url="socks5://127.0.0.1:7890",
            )
        )

        with mock.patch.dict(sys.modules, {"socks": None}):
            with self.assertRaises(LmBashError):
                client.request_command("show pwd")

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
            client.request_command("show current directory")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_http_error_with_empty_body_includes_request_context(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:1234/v1/chat/completions",
            502,
            "Bad Gateway",
            {},
            None,
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="http://localhost:1234/v1",
                api_key="",
                model="local-model",
            )
        )

        with self.assertRaisesRegex(
            LmBashError,
            "Provider returned HTTP 502 Bad Gateway from "
            "http://localhost:1234/v1/chat/completions",
        ):
            client.request_command("show current directory")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_http_error_with_unreadable_empty_body_includes_request_context(self, urlopen):
        class UnreadableBodyHTTPError(urllib.error.HTTPError):
            def read(self):
                raise KeyError("file")

        urlopen.side_effect = UnreadableBodyHTTPError(
            "http://localhost:1234/v1/chat/completions",
            502,
            "Bad Gateway",
            {},
            None,
        )
        client = build_client(
            ProviderConfig(
                provider="openai-compatible",
                base_url="http://localhost:1234/v1",
                api_key="",
                model="local-model",
            )
        )

        with self.assertRaisesRegex(
            LmBashError,
            "Provider returned HTTP 502 Bad Gateway from "
            "http://localhost:1234/v1/chat/completions",
        ):
            client.request_command("show current directory")

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

        command = client.request_command("list files")

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

        command = client.request_command("list files")

        self.assertEqual(command, "ls -la")

    @mock.patch("lmbash.providers.urllib.request.urlopen")
    def test_claude_uses_last_non_empty_line_from_multi_command_response(self, urlopen):
        urlopen.return_value = self.make_response(
            {"content": [{"type": "text", "text": "ls\nls -a"}]}
        )
        client = build_client(
            ProviderConfig(
                provider="claude-compatible",
                base_url="https://claude.example.test",
                api_key="claude-key",
                model="claude-model",
            )
        )

        command = client.request_command("list hidden files")

        self.assertEqual(command, "ls -a")

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
            client.request_command("list files")

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
