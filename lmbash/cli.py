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
from lmbash import terminal


def build_refinement_prompt(original_prompt, previous_command, edit_request):
    return (
        "Update the bash command using this context.\n\n"
        f"Original request:\n{original_prompt}\n\n"
        f"Current command:\n{previous_command}\n\n"
        f"New requirement:\n{edit_request}\n\n"
        "Return exactly one updated bash command and nothing else."
    )


def build_risk_prompt(command):
    return f"Assess this shell command. Return exactly low, medium, or high on the first line and one concise reason on the second. Do not execute it.\n\n{command}"


def build_explanation_prompt(command, detail):
    instruction = "Give three concise sections: purpose, key parts, and cautions." if detail == "brief" else "Explain each pipeline, redirect, variable, subcommand, and important option."
    return f"{instruction} Do not execute, modify, or regenerate this command.\n\n{command}"


def parse_risk_assessment(response):
    if not isinstance(response, str):
        return None
    lines = response.splitlines()
    if not lines:
        return None
    level = lines[0].strip().lower()
    reason = " ".join(line.strip() for line in lines[1:] if line.strip())
    if level not in {"low", "medium", "high"} or not reason:
        return None
    return level, reason


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
    if argv and argv[0] == "shell-init":
        parser = argparse.ArgumentParser(description="Print shell integration for lmbash")
        parser.add_argument("shell", choices=("bash", "zsh"))
        args = parser.parse_args(argv[1:])
        args.command = "shell-init"
        return args

    parser = argparse.ArgumentParser(
        description="Generate and optionally run a bash command using a configured provider.",
        epilog="Commands: config  Configure lmbash",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help=argparse.SUPPRESS,
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


def prompt_from_args(args, stream=None):
    if args.prompt:
        return " ".join(args.prompt).strip()
    if stream is None:
        return input(terminal.prompt("Describe the bash command you want: ")).strip()
    print(terminal.prompt("Describe the bash command you want: ", stream), end="", file=stream, flush=True)
    return input().strip()


def choose_action(explain_more_available=False, stream=None):
    suffix = ", [xx] explain more" if explain_more_available else ""
    message = f"Action? [y] execute, [e] edit, [x] explain{suffix}, [N] cancel "
    if stream is None:
        answer = input(terminal.action(message)).strip().lower()
    else:
        print(terminal.action(message, stream), end="", file=stream, flush=True)
        answer = input().strip().lower()
    if answer in {"y", "yes"}:
        return "execute"
    if answer in {"e", "edit"}:
        return "edit"
    if answer == "x":
        return "explain"
    if answer == "xx" and explain_more_available:
        return "explain-more"
    return "cancel"


def run_command(command):
    return subprocess.run(command, shell=True, text=True, capture_output=True)


def build_shell_init(shell):
    history_command = 'history -s "$command"' if shell == "bash" else 'print -s -- "$command"'
    return f'''lmbash() {{
  local command
  command="$(command lmbash --print-command "$@")" || return
  [ -n "$command" ] || return 0
  {history_command}
  eval "$command"
}}'''


def print_result(result):
    if result.stdout:
        print(terminal.heading("stdout"))
        print(result.stdout, end="")
    if result.stderr:
        print(terminal.heading("stderr"), file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
    print(f"\n{terminal.status(f'Exit code: {result.returncode}')}")


def load_effective_config(args):
    config = load_config()
    if config is None:
        print(terminal.status("No lmbash config found. Starting setup."))
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
            print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
            return 1
        if config is None:
            print(terminal.status("No lmbash config found."))
            return 1
        print(format_config(config))
        return 0

    if args.reset:
        try:
            removed = remove_config()
        except OSError as exc:
            print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
            return 1
        if removed:
            print(terminal.status("Config removed."))
            return 0
        print(terminal.status("No lmbash config found."))
        return 0

    try:
        config = configure_interactively()
    except ConfigError as exc:
        print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
        return 2

    try:
        save_config(config)
    except (ConfigError, OSError) as exc:
        print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
        return 1
    print(terminal.status("Config saved."))
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.command == "config":
        return handle_config_command(args)
    if args.command == "shell-init":
        print(build_shell_init(args.shell))
        return 0

    display_stream = sys.stderr if args.print_command else None
    prompt = prompt_from_args(args, stream=display_stream)
    if not prompt:
        print(terminal.error("prompt cannot be empty", stream=sys.stderr), file=sys.stderr)
        return 2

    try:
        config = load_effective_config(args)
        client = build_client(config)
    except (ConfigError, LmBashError, OSError) as exc:
        print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
        return 1

    request_prompt = prompt
    brief_explanation_seen = False
    while True:
        try:
            command = client.request_command(request_prompt)
        except LmBashError as exc:
            print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
            return 1

        print(terminal.command_block(command, stream=display_stream), file=display_stream)
        print(terminal.separator(stream=display_stream), file=display_stream)
        try:
            risk = parse_risk_assessment(client.request_analysis(build_risk_prompt(command)))
            if risk and risk[0] != "low":
                print(terminal.status(f"Risk ({risk[0]}): {risk[1]}", display_stream), file=display_stream)
        except LmBashError:
            print(terminal.status("Risk assessment unavailable", display_stream), file=display_stream)

        while True:
            action = choose_action(brief_explanation_seen, stream=display_stream)
            if action == "execute":
                if args.print_command:
                    print(command)
                    return 0
                result = run_command(command)
                print_result(result)
                return result.returncode
            if action == "cancel":
                print("Cancelled.", file=display_stream)
                return 0
            if action in {"explain", "explain-more"}:
                detail = "brief" if action == "explain" else "detailed"
                try:
                    print(client.request_analysis(build_explanation_prompt(command, detail)), file=display_stream)
                    brief_explanation_seen = True
                except LmBashError as exc:
                    print(terminal.error(str(exc), stream=sys.stderr), file=sys.stderr)
                continue
            break

        if display_stream is None:
            edit_request = input(terminal.prompt("How should the command change? ")).strip()
        else:
            print(terminal.prompt("How should the command change? ", display_stream), end="", file=display_stream, flush=True)
            edit_request = input().strip()
        if not edit_request:
            print("Cancelled.", file=display_stream)
            return 0
        request_prompt = build_refinement_prompt(prompt, command, edit_request)
        brief_explanation_seen = False


if __name__ == "__main__":
    raise SystemExit(main())
