from dataclasses import dataclass
import importlib
import json
import socket
from urllib.parse import urlparse
import urllib.error
import urllib.request


SYSTEM_PROMPT = (
    "You convert user requests into exactly one bash command. "
    "Return only the command. Do not include explanations, Markdown, or comments."
)


class LmBashError(Exception):
    pass


@dataclass
class ProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    preset: str = "custom"
    proxy_url: str = ""


def post_json(url, payload, headers, proxy_url=""):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    return open_json_request(request, proxy_url)


def open_json_request(request, proxy_url=""):
    opener = None
    original_socket = None
    if proxy_url:
        parsed = urlparse(proxy_url)
        if parsed.scheme in {"http", "https"}:
            handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            opener = urllib.request.build_opener(handler)
        elif parsed.scheme in {"socks5", "socks5h"}:
            if not parsed.hostname or not parsed.port:
                raise LmBashError("SOCKS proxy URL must include host and port")
            try:
                socks = importlib.import_module("socks")
            except ImportError as exc:
                raise LmBashError("SOCKS proxy requires PySocks to be installed") from exc
            socks.set_default_proxy(
                socks.SOCKS5,
                parsed.hostname,
                parsed.port,
                rdns=parsed.scheme == "socks5h",
            )
            original_socket = socket.socket
            socket.socket = socks.socksocket
        else:
            raise LmBashError(f"Unsupported proxy scheme: {parsed.scheme}")

    try:
        if opener is None:
            response_context = urllib.request.urlopen(request, timeout=60)
        else:
            response_context = opener.open(request, timeout=60)
        with response_context as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = format_http_error(exc)
        raise LmBashError(detail) from exc
    except urllib.error.URLError as exc:
        raise LmBashError(f"Cannot reach provider: {exc.reason}") from exc
    finally:
        if original_socket is not None:
            socket.socket = original_socket

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LmBashError("Provider returned an invalid JSON response") from exc


def format_http_error(error):
    try:
        raw_body = error.read()
    except (AttributeError, OSError, ValueError, KeyError):
        raw_body = b""
    if isinstance(raw_body, bytes):
        body = raw_body.decode("utf-8", errors="replace").strip()
    else:
        body = str(raw_body or "").strip()
    reason = str(error.reason).strip() if error.reason else ""
    reason_text = f" {reason}" if reason else ""
    url_text = f" from {error.url}" if error.url else ""
    if body:
        return f"Provider returned HTTP {error.code}{reason_text}{url_text}: {body}"
    return f"Provider returned HTTP {error.code}{reason_text}{url_text}: response body was empty"


def clean_command(content):
    command = content.strip()
    if command.startswith("```") and command.endswith("```"):
        lines = command.splitlines()
        if len(lines) >= 2:
            command = "\n".join(lines[1:-1]).strip()
    lines = [line.strip() for line in command.splitlines() if line.strip()]
    if len(lines) > 1:
        command = lines[-1]
    if not command:
        raise LmBashError("Provider returned an empty command")
    return command


class OpenAICompatibleClient:
    def __init__(self, config):
        self.config = config

    def request_command(self, prompt):
        return self.complete_command(prompt)

    def complete_command(self, prompt):
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        response = post_json(
            self.config.base_url.rstrip("/") + "/chat/completions",
            payload,
            headers,
            self.config.proxy_url,
        )

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LmBashError("Provider returned an invalid chat completion response") from exc
        if not isinstance(content, str):
            raise LmBashError("Provider returned an invalid chat completion response")
        return clean_command(content)


class ClaudeCompatibleClient:
    def __init__(self, config):
        self.config = config

    def request_command(self, prompt):
        return self.complete_command(prompt)

    def complete_command(self, prompt):
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = post_json(
            self.config.base_url.rstrip("/") + "/v1/messages",
            payload,
            {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
            self.config.proxy_url,
        )

        try:
            content_items = response["content"]
        except (KeyError, TypeError) as exc:
            raise LmBashError("Provider returned an invalid Claude message response") from exc
        if not isinstance(content_items, list):
            raise LmBashError("Provider returned an invalid Claude message response")
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    return clean_command(text)
        raise LmBashError("Provider returned an invalid Claude message response")


def build_client(config):
    if config.provider == "openai-compatible":
        return OpenAICompatibleClient(config)
    if config.provider == "claude-compatible":
        return ClaudeCompatibleClient(config)
    raise LmBashError(f"Unknown provider: {config.provider}")
