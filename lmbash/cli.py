import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from lmbash.providers import LmBashError


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"


def default_model():
    return os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)


def clean_command(content):
    command = content.strip()
    if command.startswith("```") and command.endswith("```"):
        lines = command.splitlines()
        if len(lines) >= 2:
            command = "\n".join(lines[1:-1]).strip()
    return command


def build_payload(prompt, model):
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You convert user requests into exactly one bash command. "
                    "Return only the command. Do not include explanations, Markdown, or comments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }


def build_refinement_prompt(original_prompt, previous_command, edit_request):
    return (
        "Update the bash command using this context.\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Current command:\n{previous_command}\n\n"
        f"New requirement:\n{edit_request}\n\n"
        "Return exactly one updated bash command and nothing else."
    )


def request_command(prompt, base_url, model):
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(build_payload(prompt, model)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LmBashError(f"LM Studio returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LmBashError(f"Cannot reach LM Studio: {exc.reason}") from exc

    try:
        payload = json.loads(body)
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LmBashError("LM Studio returned an invalid chat completion response") from exc

    command = clean_command(content)
    if not command:
        raise LmBashError("LM Studio returned an empty command")
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a bash command using local LM Studio."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio API base URL, default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--model", default=default_model(), help="LM Studio model name")
    parser.add_argument("prompt", nargs="*", help="Natural-language command request")
    return parser.parse_args(argv)


def prompt_from_args(args):
    if args.prompt:
        return " ".join(args.prompt).strip()
    return input("Describe the bash command you want: ").strip()


def choose_action():
    answer = input("Action? [y] execute, [e] edit, [N] cancel ").strip().lower()
    if answer in {"y", "yes"}:
        return "execute"
    if answer in {"e", "edit"}:
        return "edit"
    return "cancel"


def run_command(command):
    return subprocess.run(command, shell=True, text=True, capture_output=True)


def print_result(result):
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(f"\nExit code: {result.returncode}")


def main(argv=None):
    args = parse_args(argv)
    prompt = prompt_from_args(args)
    if not prompt:
        print("Error: prompt cannot be empty", file=sys.stderr)
        return 2

    request_prompt = prompt

    while True:
        try:
            command = request_command(request_prompt, args.base_url, args.model)
        except LmBashError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print("\nGenerated command:")
        print(command)

        action = choose_action()
        if action == "execute":
            break
        if action == "cancel":
            print("Cancelled.")
            return 0

        edit_request = input("How should the command change? ").strip()
        if not edit_request:
            print("Cancelled.")
            return 0
        request_prompt = build_refinement_prompt(prompt, command, edit_request)

    result = run_command(command)
    print_result(result)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
