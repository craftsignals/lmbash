# lmbash

Generate bash commands from natural-language requests using an LLM provider.
`lmbash` always shows the generated command first and only runs it after
explicit confirmation.

## Installation

```bash
pipx install lmbash
```

Or install with `pip`:

```bash
python3 -m pip install lmbash
```

## Requirements

- Python 3.9 or newer
- An LLM provider endpoint and model

## Providers

`lmbash` supports these provider protocols:

- `openai-compatible`: OpenAI, OpenRouter, LM Studio, Ollama, and compatible
  chat completions servers.
- `claude-compatible`: Anthropic Claude and compatible Messages API servers.

Common base URLs:

- OpenAI: `https://api.openai.com/v1`
- OpenRouter: `https://openrouter.ai/api/v1`
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- Anthropic: `https://api.anthropic.com`

## Configuration

Run the setup wizard:

```bash
lmbash config
```

Show the saved configuration:

```bash
lmbash config --show
```

Reset the saved configuration:

```bash
lmbash config --reset
```

Configuration is stored at:

```text
~/.config/lmbash/config.json
```

API keys are stored locally in that file. The config file is written with
`0600` permissions where supported, and `lmbash config --show` masks API keys.

You can override saved configuration with environment variables:

```bash
export LMBASH_PROVIDER="openai-compatible"
export LMBASH_BASE_URL="https://openrouter.ai/api/v1"
export LMBASH_API_KEY="your-api-key"
export LMBASH_MODEL="your/model-name"
export LMBASH_PROXY_URL="socks5h://127.0.0.1:7890"
```

The setup wizard can also save an optional proxy URL. Supported proxy schemes:

- `http://...`
- `https://...`
- `socks5://...`
- `socks5h://...`

For SOCKS proxies, `socks5h://127.0.0.1:7890` is usually the best choice
because DNS resolution also goes through the proxy.

## Usage

Pass the request as arguments:

```bash
lmbash list files sorted by size
```

If no config exists, `lmbash` starts the setup wizard before generating a
command.

Or run it without arguments and enter the request interactively:

```bash
lmbash
```

`lmbash` prints the generated command and asks what to do:

```text
Generated command:
ls -lhS
Action? [y] execute, [e] edit, [N] cancel
```

Choose:

- `y` or `yes` to execute
- `e` or `edit` to ask for a revised command
- Enter, `n`, or anything else to cancel

Override the provider API URL:

```bash
lmbash --base-url http://localhost:1234/v1 show current directory
```

Override the model:

```bash
lmbash --model your/model-name show current directory
```

## Safety

`lmbash` generates shell commands with an LLM. Review every command before
executing it. Do not run commands you do not understand, especially commands
that delete files, change permissions, install software, or send data over the
network.

## Development

Run tests:

```bash
python3 -m unittest -v
```

Build distributions:

```bash
python3 -m build
```
