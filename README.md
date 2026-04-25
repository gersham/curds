# curds

> _to complement your fries and gravy_

Generate images from the command line via OpenAI's gpt-image-2 (direct) or
Replicate-hosted models. Logs upstream progress in colorized
[logfmt](https://brandur.org/logfmt). When prompt or token is missing
curds clears the screen and drops into a Bubble Tea TUI with a CURDS
banner, multiline prompt, spinner, scrolling log panel, and a "generate
another?" loop after each render.

## Requirements

- **Go 1.26+** for building (curds uses generics, structured logging, and
  recent stdlib helpers; older toolchains will refuse to compile).
  Install via [go.dev/dl](https://go.dev/dl/), Homebrew (`brew install go`),
  asdf, or your distro's package manager.
- **macOS, Linux, or Windows.** `-open` uses `open` on macOS,
  `xdg-open` on Linux, and `cmd /C start` on Windows.
- A terminal that supports ANSI colors and Unicode box drawing for the
  best TUI experience (any modern terminal qualifies — iTerm2, Alacritty,
  Kitty, Windows Terminal, GNOME Terminal, etc.).
- An API token from at least one provider:
  - [OpenAI](https://platform.openai.com/api-keys) — needs API
    Organization Verification to call gpt-image-2.
  - [Replicate](https://replicate.com/account/api-tokens) — alternative
    backend, no verification dance.
- Network access to `api.openai.com` and/or `api.replicate.com`
  (and `replicate.delivery` for downloading rendered images).

## Install

```bash
./install.sh                  # builds and installs to /usr/local/bin or ~/.local/bin
```

Or build manually:

```bash
go build -o curds ./cmd/curds
```

## First run

```bash
curds -prompt "a watercolor fox in a meadow"
```

Or with no flags at all to land in the TUI. On first run curds writes a
default config to `~/.config/curds/config.toml` and saves output to
`~/Desktop/curds/<unix_milli>.webp`.

## Interactive mode (TUI)

Triggered when prompt or token is missing (suppress with `-no-tui`).

- Clears the screen and renders the CURDS banner.
- Captures provider+token via a one-shot form when none is configured;
  optionally writes the token back to the config file.
- Multiline prompt input. `ctrl+d` submits, `ctrl+c` quits.
- During generation: spinner + a bordered, scrolling log panel showing
  every upstream event live (`generation.started`, `openai.request`,
  `image.received`, …).
- After generation: shows the saved path(s) and elapsed time.
  Press `g` to clear the prompt and generate again, `q` to quit.

## Tokens

Resolution priority (first non-empty wins):

1. `-token` flag
2. `[tokens]` section of `~/.config/curds/config.toml`
3. `.env` in the current working directory
4. process environment (`OPENAI_API_KEY`, `REPLICATE_API_TOKEN`)

If no token is found at startup, curds opens the TUI and offers to capture
one and (optionally) save it to the config.

## Providers

curds auto-selects the provider based on which token is available, with
OpenAI preferred when both are set. Override with `-provider openai|replicate`
or by setting `provider = "openai"` in the config file.

| Provider  | Default model         | Endpoint                                                     |
|-----------|-----------------------|--------------------------------------------------------------|
| openai    | `gpt-image-2`         | `/v1/images/generations` (or `/v1/images/edits` with `-i`)   |
| replicate | `openai/gpt-image-2`  | `/v1/models/<owner>/<name>/predictions` (sync via `Prefer: wait`) |

## Editing / composing with reference images

Pass `-input-image` (repeatable or comma-separated, up to 16). curds
switches to OpenAI's `/v1/images/edits` endpoint or Replicate's
`input_images` field automatically.

```bash
# Compose from two references
curds -input-image body-lotion.png,bath-bomb.png \
      -prompt "Relax & Unwind gift basket"

# Mask-driven inpainting (OpenAI only; mask must be PNG with alpha)
curds -input-image lounge.png -mask mask.png \
      -prompt "indoor lounge with flamingo in pool"
```

## Aspect ratios

`-aspect-ratio` accepts these named ratios (mapped to multiples-of-16
sizes for gpt-image-2):

| Ratio       | OpenAI size  | Notes                          |
|-------------|--------------|--------------------------------|
| 1:1         | 1024×1024    |                                |
| 3:2 / 2:3   | 1536×1024 / 1024×1536 |                       |
| 4:3 / 3:4   | 1536×1152 / 1152×1536 |                       |
| **16:9**    | **2048×1152** | default; ~1080p+ landscape    |
| 9:16        | 1152×2048    | ~1080p+ portrait               |
| 21:9 / 9:21 | 2688×1152 / 1152×2688 | ultrawide              |
| 2:1 / 1:2   | 2048×1024 / 1024×2048 |                       |
| 16:9-4k / 9:16-4k | 3840×2160 / 2160×3840 |                |

Replicate's gpt-image-2 wrapper accepts only `1:1`, `3:2`, `2:3`.

For something custom, pass `-size WxH`. Anything not on a 16-pixel
boundary is rounded to the nearest valid value (true 1080p `1920×1080`
becomes `1920×1088`, for example).

## Configuration

`~/.config/curds/config.toml` (auto-created):

```toml
provider = ""                   # "openai", "replicate", or "" to auto-detect
default_model = "gpt-image-2"

[output]
directory = "~/Desktop/curds"
format = "webp"
compression = 90

[tokens]
openai = ""
replicate = ""

[defaults]
quality = "auto"
aspect_ratio = "16:9"
background = "auto"
moderation = "auto"
number_of_images = 1

[models.gpt-image-2]
openai_name = "gpt-image-2"
replicate_name = "openai/gpt-image-2"
```

Override the path with `$CURDS_CONFIG`.

## Logging

Every stage emits a logfmt event to stderr. On a TTY events are colorized;
piped output stays plain logfmt for log collectors. Add `-v` for
debug-level events (request bodies, etc.).

```
ts=2026-04-25T17:34:00.123Z level=info  event=curds.start version=0.1.0
ts=2026-04-25T17:34:00.140Z level=info  event=config.loaded path=…
ts=2026-04-25T17:34:00.180Z level=info  event=generation.started provider=openai size=2048x1152 prompt_chars=42
ts=2026-04-25T17:34:00.181Z level=info  event=openai.request endpoint=… kind=generations
ts=2026-04-25T17:34:08.420Z level=info  event=openai.response status_code=200 bytes=987432
ts=2026-04-25T17:34:08.421Z level=info  event=image.received index=0 bytes=987432 format=webp
ts=2026-04-25T17:34:08.435Z level=info  event=curds.completed images=1 duration_ms=8312 paths=…
```

Upstream errors are captured at `level=error`:

```
ts=… level=error event=openai.api_error status_code=400 body="{ \"error\": { … } }"
```

## CLI flags

Run `curds -h` for the full list. Highlights:

- `-prompt` — prompt text (or pipe via stdin, or use the TUI)
- `-output` — output path (default: `~/Desktop/curds/<ms>.<format>`)
- `-aspect-ratio` / `-size` — image dimensions
- `-quality` / `-output-format` / `-output-compression`
- `-input-image` — reference image(s), repeatable or comma-separated
- `-mask` — mask file for OpenAI edits
- `-provider`, `-token`, `-model`
- `-open` — open generated images in OS viewer (macOS Preview)
- `-verbose` — debug-level logs
- `-no-tui` — never enter interactive mode (fail instead)

## Layout

```
.
├── cmd/curds/main.go      package main — CLI shell
├── client.go              package curds — Client, Request, Result, providers
├── openai.go              ↑ OpenAI Image API impl
├── replicate.go           ↑ Replicate API impl
├── log.go                 ↑ shared logfmt + lipgloss formatter
├── curds_test.go          ↑ unit + httptest integration tests
├── config/                package config — TOML, .env, token resolution
├── tui/                   package tui  — huh-driven interactive form
├── install.sh             builds + installs to /usr/local/bin or ~/.local/bin
└── go.mod                 module github.com/gersham/curds
```

## License

Personal project. No license attached yet.
