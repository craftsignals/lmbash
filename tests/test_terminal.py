import os
import unittest
from unittest import mock

from lmbash import terminal


class FakeStream:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


class TerminalTests(unittest.TestCase):
    def test_style_plain_for_non_tty(self):
        self.assertEqual(terminal.style("Error", "red", stream=FakeStream(False)), "Error")

    def test_style_plain_when_no_color_is_set(self):
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(terminal.style("Error", "red", stream=FakeStream(True)), "Error")

    def test_style_uses_ansi_for_tty(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(terminal.style("Error", "red", stream=FakeStream(True)), "\033[31mError\033[0m")

    def test_heading_contains_rule_and_title(self):
        text = terminal.heading("Generated command", stream=FakeStream(False))
        self.assertIn("Generated command", text)
        self.assertIn("─", text)

    def test_command_block_contains_command(self):
        text = terminal.command_block("ls -la", stream=FakeStream(False))
        self.assertIn("Generated command", text)
        self.assertIn("ls -la", text)


if __name__ == "__main__":
    unittest.main()
