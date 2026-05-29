import argparse
from dataclasses import replace
import subprocess
import sys

from lmbash.config import (
    ConfigError,
    apply_env_overrides,
    configure_interactively,
    format_config,
    load_config,
    remove_config,
    save_config,
)
from lmbash.providers import LmBashError, build_client


def clean_command(content):
    command = content.strip()
    if command.startswith("```") and command.endswith("```"):
        lines = command.splitlines()
        if len(lines) >= 2:
            command = "\n".join(lines[1:-1]).strip()
    return command


def build_refinement_prompt(original_prompt, previous_command, edit_request):
    return (
        "Update the bash command using this context.\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Current command:\n{previous_command}\n\n"
        f"New requirement:\n{edit_request}\n\n"
        "Return exactly one updated bash command and nothing else."
    )


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "config":
        parser = argparse.ArgumentParser(description="Manage lmbash config")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--show", action="store_true", help="Show saved config")
        group.add_argument("--reset", action="store_true", help="Remove saved config")
        args = parser.parse_args(argv[1:])
        args.command = "config"
        return args

    parser = argparse.ArgumentParser(
        description="Generate and optionally run a bash command using a configured provider."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Provider API base URL override",
    )
    parser.add_argument("--model", default=None, help="Provider model name override")
    parser.add_argument("prompt", nargs="*", help="Natural-language command request")
    args = parser.parse_args(argv)
    args.command = None
    return args


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


def load_effective_config(args):
    config = load_config()
    if config is None:
        print("No lmbash config found. Starting setup.")
        config = configure_interactively()
        save_config(config)

    config = apply_env_overrides(config)
    if args.base_url is not None:
        config = replace(config, base_url=args.base_url)
    if args.model is not None:
        config = replace(config, model=args.model)
    return config


def handle_config_command(args):
    if args.show:
        try:
            config = load_config()
        except ConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if config is None:
            print("No lmbash config found.")
            return 1
        print(format_config(config))
        return 0

    if args.reset:
        try:
            removed = remove_config()
        except OSError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if removed:
            print("Config removed.")
            return 0
        print("No lmbash config found.")
        return 0

    try:
        config = configure_interactively()
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        save_config(config)
    except (ConfigError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Config saved.")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.command == "config":
        return handle_config_command(args)

    prompt = prompt_from_args(args)
    if not prompt:
        print("Error: prompt cannot be empty", file=sys.stderr)
        return 2

    try:
        config = load_effective_config(args)
        client = build_client(config)
    except (ConfigError, LmBashError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    request_prompt = prompt

    while True:
        try:
            command = client.request_command(request_prompt)
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
