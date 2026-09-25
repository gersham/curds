# curds — agent notes

This file is read by Claude Code (and `AGENTS.md`-aware tools) as
project-local context. Keep it short and load-bearing.

## What this repo is

A Go CLI + library for generating images via OpenAI's gpt-image-2.5 (direct;
Flare default, Sunburst via `-model gpt-image-2.5-sunburst`),
plus images/videos via Replicate-hosted models (FLUX.2 [pro], Nano Banana 2,
Seedance 2.5, Kling 3.0, MiniMax H3, Grok Imagine Video 1.5), music and sound
effects via `elevenlabs/music` (-model music), `minimax/music-2.6`
(-model music-vocal, alias minimax-music) and `stability-ai/stable-audio-2.5`
(-model sfx), text to speech via `google/gemini-3.1-flash-tts` (-model tts),
`minimax/speech-2.8-hd` (-model tts-minimax), `elevenlabs/v3`
(-model tts-elevenlabs) and OpenAI's `gpt-4o-mini-tts` (-model tts-openai;
`tts-1-hd` also selectable), talking heads and
lip-sync via `kwaivgi/kling-avatar-v2` and `sync/lipsync-2-pro`, plus background
removal via `bria/remove-background` (segmentation), plus image upscaling via
`prunaai/p-image-upscale` (-model upscale), `nightmareai/real-esrgan`
(-model upscale-esrgan) and `topazlabs/image-upscale` (-model upscale-pro).
`curds run OWNER/MODEL key=value …` is a raw Replicate passthrough (`run.go` in
the library, `cmd/curds/run.go` in the CLI); it shares `ReplicateProvider`'s
prediction lifecycle (`runPrediction`) but does no field mapping.
Module path: `github.com/gersham/curds`.

## Layout

```
.
├── cmd/curds/main.go      package main — CLI flag parsing, TUI dispatch, output
├── cmd/curds/run.go       package main — the `curds run` subcommand
├── client.go              package curds — Client, Request, Result, types
├── openai.go              package curds — OpenAI Image API provider
├── replicate.go           package curds — Replicate API provider (incl. the
│                          Seedance→Kling face-rejection fallback)
├── run.go                 package curds — raw passthrough Run/Schema/input parsing
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
- **Video models.** `seedance-2.5` (`bytedance/seedance-2.5`) is the default
  video model when output is MP4 and no `-model` is supplied; the CLI falls
  back to native xAI `grok-imagine-video` only when no replicate token is
  available. `seedance-2` (`bytedance/seedance-2.0`) stays selectable and is
  the only Seedance with 1080p. `kling-v3` (`kwaivgi/kling-v3-video`),
  `minimax-h3` (`minimax/h3`) and `grok-imagine-video-1.5` are selectable. All
  emit `Result.Videos` and save MP4 output.
  The two Seedance versions differ, so validation is model-aware:
  `SeedanceLimitsFor` / `SeedanceAspectRatiosFor` return 2.5's wider
  ceilings (30 reference images, 10 videos, 10 audios, 4-30s, 480p/720p, no
  `9:21`) or 2.0's (9/3/3, 4-15s, 480p/720p/1080p, `9:21`). 1080p on 2.5 is a
  usage error (exit 2) naming `-model seedance-2` / `-model kling-v3`; the
  library rejects it too. `MaxInputImagesFor` raises the generic
  `-input-image` cap for Seedance (first frame + model-aware references).
  Kling maps `-video-resolution` onto its `mode` enum via `KlingMode`
  (720p=standard, 1080p=pro, 4k) and names frames `start_image`/`end_image`.
  MiniMax H3 has its own field names (`first_frame_image`, `last_frame_image`,
  `reference_image_urls`/`_video_`/`_audio_`, `ratio`) and its own enums —
  resolution `768P`/`2K` (mapped from `768p`/`2k` by `MinimaxVideoResolution`),
  ratio with `adaptive` and no `auto`, duration 4-15s. No seed, no audio toggle.
  Talking heads: `kling-avatar` (`kwaivgi/kling-avatar-v2`) sends exactly one
  `image` + `audio` + `mode` (720p/std, 1080p/pro; default pro) and omits an
  empty prompt; `lipsync` (`sync/lipsync-2-pro`) sends `video` + `audio` +
  `active_speaker` + optional `sync_mode`/`temperature` and takes no prompt.
  `IsPromptlessModel` is what exempts segmentation/upscale/lipsync/kling-avatar
  from the "prompt is required" check — extend it, not the call sites.
  `Request.Audio`/`InputVideo`/`SyncMode`/`SyncTemperature` carry their flags;
  `-strip-audio` defaults off for both models and `maybeCropCaptions` handles
  `-crop-captions`. Seedance failures mentioning E005/"flagged as sensitive"
  retry once on Kling 3.0 (`fallbackOnSeedanceFaceRejection`, `-no-fallback`),
  for both Seedance versions.
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
- **Upscale models.** `IsUpscaleModel` covers three different contracts.
  `upscale` (`prunaai/p-image-upscale`, the 2026 default) sends
  `upscale_mode=factor` with `factor` from `-scale` (1-8, default 4) and
  `output_format` png/jpg/webp (curds' `jpeg` → `jpg`); it has no
  face-enhancement knob, so `-face-enhance` is a usage error (exit 2) and the
  error names `upscale-pro` / `upscale-esrgan`. `upscale-esrgan`
  (`nightmareai/real-esrgan`, 2021-era) keeps the numeric `scale` (1-10) plus
  GFPGAN `face_enhance`, and returns PNG. `upscale-pro`
  (`topazlabs/image-upscale`) maps `-scale` onto the `upscale_factor` enum via
  `TopazUpscaleFactor`. `buildReplicateUpscaleInput` branches on the same
  predicates; `PrunaUpscaleFormats` is what `validateUpscale` checks. Changing
  which model `-model upscale` means requires updating `DefaultUpscaleModel`,
  `config.builtinModels`, `DefaultTOML`, and the README/help at once.
- **Text-to-speech models.** `tts` (`google/gemini-3.1-flash-tts`, the
  default TTS model), `tts-minimax` (`minimax/speech-2.8-hd`),
  `tts-elevenlabs` (`elevenlabs/v3`) and `tts-openai` /
  `tts-1-hd` (OpenAI `gpt-4o-mini-tts` / `tts-1-hd`, via the openai provider's
  `POST /v1/audio/speech`, returning raw bytes). They are audio models: they
  emit `Result.Audios`, save mp3/wav (`saveAudios`/`writeAudioAsset`), and are
  covered by `IsAudioModel`. `IsTTSModel` plus
  `IsTTSGeminiModel`/`IsTTSSpeechModel`/`IsTTSElevenLabsModel`/
  `IsOpenAITTSModel`/`IsOpenAITTSMiniModel` gate the per-model validation
  (`validateTTS` in client.go, `validateTTSFlags` in the CLI) and builders
  (`buildReplicateGeminiTTSInput`, `buildReplicateSpeechInput`,
  `buildReplicateElevenLabsTTSInput`, `OpenAIProvider.callSpeech`). Replicate
  TTS needs a replicate token, OpenAI TTS an openai token; a config model
  bound to one provider forces it, and incompatible `-provider`/`-model` pairs
  are rejected locally. Flags: `-voice` (free-form for tts-minimax, enum for
  the other three), `-emotion`/`-pitch` (tts-minimax),
  `-stability`/`-style` (tts-elevenlabs), `-instructions` (tts → Gemini's
  `prompt` style field, and tts-openai; rejected elsewhere), `-speed`
  (tts-minimax 0.5-2, tts-elevenlabs 0.7-1.2, tts-openai 0.25-4 — Gemini has
  no speed knob). `Request.Stability`/`Style` are `*float64` (nil = model
  default) because 0 is a valid value. Min/max text lengths are checked
  locally (10000 chars / 4096 chars / 4000 bytes for Gemini text and prompt).
  Gemini returns WAV and ElevenLabs mp3, so a mismatched container is
  transcoded after download rather than requested upstream. elevenlabs/v3 on
  Replicate does NOT parse inline markers: bracketed text is spoken aloud, and
  no docs may claim otherwise.
- **Audio models.** `music` (`elevenlabs/music`), `music-vocal`
  (`minimax/music-2.6`, alias `minimax-music`) and `sfx`
  (`stability-ai/stable-audio-2.5`) are Replicate-only, prompt-driven (except
  that music-vocal also runs on `-lyrics` alone), and emit `Result.Audios` +
  `mp3`/`wav` output. `IsAudioModel` gates their validation,
  request-building, download and save paths. `-duration` maps onto
  `music_length_ms` (music, 5-300s) or `duration` (sfx, 1-190s, default 10);
  music-vocal ignores length upstream, so the CLI trims post-download with a
  2s fade-out (`maybeTrimAudio`/`trimAudioInPlace`, ffmpeg-gated like
  `maybeStripAudio`). `-lyrics` (`TEXT` or `@file.txt`) is music-vocal only
  and implies vocals, which forces `is_instrumental=false`; otherwise
  `Request.Instrumental` (nil = model default) decides. Sfx picks its own
  container, so `saveAudios`/`writeAudioAsset` transcode to the requested one
  via ffmpeg when available, else keep the returned container's extension and
  log `audio.format_mismatch`. Enums: elevenlabs mp3 → `mp3_high_quality`,
  wav → `wav_cd_quality`; minimax sends `sample_rate` 44100 / `bitrate`
  256000. `IsPromptlessModel` must NOT list audio models.
- **Default output path:** `<config.output.directory>/<unix_milli>.<format>`.
  Changing the default path → update `config.DefaultTOML` and the README.
  When a model can only emit one format (segmentation, upscale, Nano Banana),
  `applyModelOutputDefaults` also retargets an explicit `-output` extension
  rather than writing PNG bytes into a `.webp` name.
- **`curds run`** is a subcommand, dispatched from `realMain` before flag
  parsing. Inputs are `key=value` (`ParseRunInput`): `@path` uploads a file as
  a data URL, everything else is JSON when it parses and a string otherwise;
  repeating a key builds an array. `-schema` prints the model document's
  `Input` properties (enums resolve through `allOf`/`$ref`).
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
