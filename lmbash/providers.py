from dataclasses import dataclass
import json
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


def post_json(url, payload, headers):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LmBashError(f"Provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LmBashError(f"Cannot reach provider: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LmBashError("Provider returned an invalid JSON response") from exc


class OpenAICompatibleClient:
    def __init__(self, config):
        self.config = config

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
        )

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LmBashError("Provider returned an invalid chat completion response") from exc
        if not isinstance(content, str):
            raise LmBashError("Provider returned an invalid chat completion response")
        return content


class ClaudeCompatibleClient:
    def __init__(self, config):
        self.config = config

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
                    return text
        raise LmBashError("Provider returned an invalid Claude message response")


def build_client(config):
    if config.provider == "openai-compatible":
        return OpenAICompatibleClient(config)
    if config.provider == "claude-compatible":
        return ClaudeCompatibleClient(config)
    raise LmBashError(f"Unknown provider: {config.provider}")
