import os
import sys


COLORS = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "bold": "1",
}

COMMAND_HEADING = "Generated command"


def supports_color(stream=None):
    stream = stream or sys.stdout
    return not os.environ.get("NO_COLOR") and hasattr(stream, "isatty") and stream.isatty() is True


def style(text, color, stream=None):
    if not supports_color(stream):
        return text
    return f"\033[{COLORS[color]}m{text}\033[0m"


def heading(text, stream=None):
    return f"{style(text, 'bold', stream)}\n{'─' * len(text)}"


def prompt(text, stream=None):
    return style(f"lmbash › {text}", "cyan", stream)


def action(text, stream=None):
    return style(text, "yellow", stream)


def status(text, stream=None):
    return style(text, "cyan", stream)


def error(text, stream=None):
    return style(f"Error: {text}", "red", stream)


def command_block(command, stream=None):
    return f"\n{heading(COMMAND_HEADING, stream)}\n{style(command, 'green', stream)}"


def separator(stream=None):
    return style("─" * len(COMMAND_HEADING), "cyan", stream)
