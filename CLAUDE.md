# curds — agent notes

This file is read by Claude Code (and `AGENTS.md`-aware tools) as
project-local context. Keep it short and load-bearing.

## What this repo is

A Go CLI + library for generating images via OpenAI's gpt-image-2.5 (direct;
Flare default, Sunburst via `-model gpt-image-2.5-sunburst`),
plus images/videos via Replicate-hosted models (FLUX.2 [pro], Nano Banana 2,
Seedance 2.0, Kling 3.0, MiniMax H3, Grok Imagine Video 1.5), plus background
removal via `bria/remove-background` (segmentation), plus image upscaling via
`nightmareai/real-esrgan` and `topazlabs/image-upscale` (super-resolution).
Module path: `github.com/gersham/curds`.

## Layout

```
.
├── cmd/curds/main.go      package main — CLI flag parsing, TUI dispatch, output
├── client.go              package curds — Client, Request, Result, types
├── openai.go              package curds — OpenAI Image API provider
├── replicate.go           package curds — Replicate API provider
├── log.go                 package curds — shared logfmt + lipgloss formatter
├── curds_test.go          package curds — unit + httptest tests
├── config/                package config — TOML loader, .env, token resolution
├── tui/                   package tui — huh-driven interactive form
├── skill/                 standalone agent skill (SKILL.md + reference.md)
└── install.sh             builds and installs to /usr/local/bin (or ~/.local/bin)
```

The library lives at the module root. The binary is in `cmd/curds`. There is
no `imagegen` package or `replicate-image-gen` directory — those were the
pre-rebrand names; the rename is complete and they should not return.

## Agent skill

`skill/` is a **standalone** agent skill (`SKILL.md` + `reference.md`) that
teaches a coding agent to drive the `curds` CLI mechanically. It is deliberately
self-contained — no `../.references/` includes — so anyone who installs curds
can install the skill by copying the directory (e.g. to `~/.claude/skills/curds`).
Agents installing curds should install this skill too, but **only if no `curds`
skill already exists** (don't clobber a marketplace-managed one). Keep `skill/`
in sync with the CLI flags and provider behaviour documented elsewhere in this
file; if you add a provider, model, or flag that changes the happy path, update
`skill/SKILL.md` (commands) and/or `skill/reference.md` (full parameter table).

## Conventions

- **Logs are logfmt.** Every event uses `curds.FormatLogLine(level, event, kv, color)`.
  Levels: `info`, `error`, `debug`. Verbose-only events use `logDebug` and
  require `Request.Verbose=true`. Don't `fmt.Fprintf` ad-hoc lines to stderr —
  it bypasses TTY-aware coloring.
- **Token handling:** `config.ResolveToken` is the single source of truth for
  the priority chain (`config > .env > env`). Don't read tokens directly from
  os.Getenv anywhere except `config/`.
- **Provider URL guards:** `isReplicateHost` parses the URL — never substring-
  match. Same idea for any future provider host check (an attacker can put
  `replicate.delivery` in a query string).
- **Verbose log redaction:** request bodies pass through `redactRequestBody`
  before logging so `openai_api_key` (BYO passthrough) never lands in stderr.
- **Aspect ratios** are mapped to multiples-of-16 sizes per gpt-image-2's
  rules. User-supplied `-size WxH` is rounded by `RoundSize`. If you expand
  the ratio map, the new entries must satisfy: both edges multiples of 16,
  edges ≤ 3840, ratio ≤ 3:1, total pixels in [655 360, 8 294 400].
- **Video models.** `seedance-2` (`bytedance/seedance-2.0`) is the default
  video model when output is MP4 and no `-model` is supplied; the CLI falls
  back to native xAI `grok-imagine-video` only when no replicate token is
  available. `kling-v3` (`kwaivgi/kling-v3-video`), `minimax-h3`
  (`minimax/h3`) and `grok-imagine-video-1.5` are selectable. All emit
  `Result.Videos` and save MP4 output.
  Kling maps `-video-resolution` onto its `mode` enum via `KlingMode`
  (720p=standard, 1080p=pro, 4k) and names frames `start_image`/`end_image`.
  MiniMax H3 has its own field names (`first_frame_image`, `last_frame_image`,
  `reference_image_urls`/`_video_`/`_audio_`, `ratio`) and its own enums —
  resolution `768P`/`2K` (mapped from `768p`/`2k` by `MinimaxVideoResolution`),
  ratio with `adaptive` and no `auto`, duration 4-15s. No seed, no audio toggle.
  For Grok Imagine Video, exactly one `InputImages` entry is sent as `image`.
  For Seedance, one `InputImages` entry is sent as `image` (first frame);
  multiple `InputImages` entries are sent as `reference_images`.
  Keep model-specific video fields on `Request` and inside
  `ReplicateProvider`; don't add video HTTP calls to the CLI or TUI.
- **Alt image models.** `flux-2-pro` and `nano-banana-2` do NOT share the
  gpt-image-2 wrapper's input shape — they get their own builders and
  validators. FLUX sizes by `resolution` (megapixels) or `width`/`height` with
  `aspect_ratio: "custom"`; Nano Banana takes `image_input` (not
  `input_images`) and 1K/2K/4K. Both map curds' `jpeg` to `jpg` via
  `replicateImageFormat`, take one image per prediction, and ignore
  quality/background/moderation. `Request.ImageResolution` carries
  `-image-resolution` for both.
- **Default output path:** `<config.output.directory>/<unix_milli>.<format>`.
  Changing the default path → update `config.DefaultTOML` and the README.
  When a model can only emit one format (segmentation, upscale, Nano Banana),
  `applyModelOutputDefaults` also retargets an explicit `-output` extension
  rather than writing PNG bytes into a `.webp` name.
- **No emojis, no chatty trailing summaries** in user-facing CLI output. Logs
  are the audit trail; stdout is just the saved file path(s).
- **One canonical name per flag.** No `-p`/`-prompt` aliases. Long forms
  only — they're self-documenting and play better with LLM tool-use.
- **TUI is bubbletea.** `tui.RunInteractive` owns the main loop (banner,
  prompt, generating, result, generate-another). The CLI passes a
  `GenerateFn` callback that builds a `curds.Request` and runs it; logs
  flow through an `io.Writer` so the TUI's panel and the CLI's stderr
  see the same logfmt events.

## Build & test

```bash
go build -o curds ./cmd/curds
go test ./...
./install.sh
```

`go test ./...` should pass before any commit. The httptest-based provider
tests are deterministic and fast.

## Adding a provider

1. New file `<name>.go` in the root (`package curds`) implementing
   `Provider` (`Generate`, `Name`).
2. Wire it in `Client.providerFor` and `New()`.
3. Add a `ProviderXXX` constant and a row in the `providerModels` table
   (default + supported list). `-provider` without `-model` reads that
   default; incompatible pairs fail in `CheckProviderModel` before any
   network call.
4. If it has different aspect-ratio constraints, gate them in `Request.Validate`.
5. Cover the happy path, an error response, and any polling/edit flow with
   httptest in `curds_test.go`.

## Adding a model

`config.toml`:

```toml
[models.<key>]
openai_name = "..."
replicate_name = "..."
```

The CLI's `-model <key>` looks the value up via `config.ResolveModel`. Add the
same entry to `config.builtinModels` so config files written before the model
existed still resolve the key instead of passing it through raw.

## Things to avoid

- Don't add a `replicate-image-gen` or `imagegen` directory — the rebrand is
  intentional and complete.
- Don't reach into `os.Getenv` for `OPENAI_API_KEY` / `REPLICATE_API_TOKEN`
  outside of `config/` — it breaks the priority chain.
- Don't write to the user's `~/.config/curds/config.toml` except via
  `config.SaveTokens` (called only by the TUI when the user opted in).
- Don't add network calls outside the providers — the CLI should never talk
  HTTP directly. Same for the TUI.

## Files outside the repo curds touches

- `~/.config/curds/config.toml` — auto-created on first run
- `~/Desktop/curds/<ms>.<format>` — default output dir, auto-created
- `.env` in cwd — read-only, optional
