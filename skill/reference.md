# Curds CLI — full parameter reference

Written from `curds --help` for version `0.2.0`; if the installed help text
differs, prefer the current `curds --help` output.

## Output formats

`-output-format` accepts `webp`, `png`, `jpeg`, or `mp4`; the output file
extension also drives the format unless the flag is set. For ordinary video
generation, `.mp4` uses the current default video model unless `-provider` or
`-model` overrides it. OpenAI supports `-output-compression` for WebP and JPEG.

## Image sizes and aspect ratios

OpenAI `gpt-image-2` named aspect ratios: `1:1`, `3:2`, `2:3`, `4:3`, `3:4`,
`16:9`, `9:16`, `21:9`, `9:21`, `2:1`, `1:2`, `16:9-4k`, `9:16-4k`.
Custom `-size WxH` is OpenAI-only and is rounded to model constraints.
Replicate's `openai/gpt-image-2` wrapper only accepts `1:1`, `3:2`, and `2:3`.

Transparent image generation is not supported by `gpt-image-2`; use
`-model remove-bg` for transparent PNG cutouts from an existing image.

## Reference images and masks

- `-input-image` accepts file paths, `http(s)` URLs, or `data:` URLs.
- Repeat `-input-image` or pass comma-separated values, up to 16 total for
  OpenAI edits.
- `-mask` is OpenAI edits only and must be a PNG with an alpha channel:

```bash
curds -no-tui -provider openai -input-image source.png -mask mask.png -prompt "$PROMPT" -output "$OUT"
```

## Seedance 2.0 video (default mp4 model)

Default Replicate video model: `bytedance/seedance-2.0` (model key
`seedance-2`). Text-to-video, first/last frame, or reference-guided.

- `-video-duration`: `-1` (intelligent) or `4`–`15` seconds (default `5`).
- `-video-resolution`: `480p`, `720p`, or `1080p` (default `720p`).
- `-aspect-ratio`: `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `21:9`, `9:21`,
  `adaptive` (default `16:9`).
- `-input-image`: one entry is the first frame; several become references.
- On request only: `-last-frame-image`, `-reference-image` (≤9),
  `-reference-video` (≤3), `-reference-audio` (≤3), `-no-audio`, `-seed`.
- Audio is stripped automatically. Do not switch models for ordinary mp4
  generation.

## Kling 3.0 video (only when the user asks for it)

Model key `kling-v3` (`kwaivgi/kling-v3-video`). Text-to-video or
image-to-video with start/end frames, native audio with lip-synced dialogue.

- `-video-duration`: `3`–`15` seconds (default `5`).
- `-video-resolution`: `720p`, `1080p`, `4k` (default `1080p`) — maps to
  Kling's standard/pro/4k mode.
- `-aspect-ratio`: `16:9`, `9:16`, `1:1` (default `16:9`).
- `-input-image`: 0 or 1 start frame; `-last-frame-image` needs one.
- No reference images/videos/audio and no `-seed`; use `seedance-2` for those.

## MiniMax H3 video (only when the user asks for it)

Model key `minimax-h3` (`minimax/h3`), Replicate. Text-to-video, or
first/last-frame image-to-video, or reference-guided.

- `-video-duration`: `4`–`15` seconds (default `5`).
- `-video-resolution`: `768p` or `2k` (default `768p`).
- `-aspect-ratio`: `adaptive`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`
  (default `16:9`; no `auto`).
- `-input-image`: 0 or 1, used as the first frame. `-last-frame-image` requires
  one `-input-image`.
- On request only: `-reference-image` (≤9), `-reference-video` (≤3),
  `-reference-audio` (≤3).
- No `-seed` and no `-no-audio`. Audio is stripped automatically.

## Seedance video (only when the user asks for it)

```bash
curds -no-tui -provider replicate -model seedance-2 -aspect-ratio 16:9 -video-duration 5 -video-resolution 720p -prompt "$PROMPT" -output "$OUT"
```

- `-video-duration`: `-1` or `4`–`15` seconds.
- `-video-resolution`: `480p`, `720p`, or `1080p`.
- Seedance-specific flags, only on request: `-no-audio`, `-last-frame-image`,
  `-reference-image`, `-reference-video`, `-reference-audio`.

## Alternative image models (only when the user asks, or for cost)

`gpt-image-2` via `-provider openai` stays the default. Two Replicate models
cover what it does poorly; both take references via `-input-image`, produce one
image per request, ignore `-quality`/`-background`/`-moderation`, and reject
`-mask`.

```bash
curds -no-tui -model flux-2-pro -aspect-ratio 16:9 -image-resolution 2mp -prompt "$PROMPT" -output "$OUT.png"
curds -no-tui -model nano-banana-2 -image-resolution 4k -input-image a.png,b.png -prompt "$PROMPT" -output "$OUT.png"
```

- `flux-2-pro`: cheap and fast (~$0.015/gen). `-image-resolution`
  `0.5mp|1mp|2mp|4mp|match_input_image`, or `-size WxH` with edges 256-2048.
  Ratios: `match_input_image`, `1:1`, `16:9`, `3:2`, `2:3`, `4:5`, `5:4`,
  `9:16`, `3:4`, `4:3`. Supports `-seed`.
- `nano-banana-2`: character consistency and multi-image compositing, cheap 4K.
  `-image-resolution` `1k|2k|4k`. PNG or JPEG only (no webp), no `-seed`,
  no `-size`.

## Background removal and upscaling (Replicate)

Both take exactly one `-input-image` (file path, `http(s)` URL, or `data:`
URL), no `-prompt`, and force PNG output; `-aspect-ratio` and `-size` are
ignored. For upscaling prefer `-model upscale-pro`
(`topazlabs/image-upscale`, `-scale` 2/4/6) — it beats `upscale`
(`nightmareai/real-esrgan`, `-scale` 1-10) on faces and text.

```bash
# Background removal → transparent PNG (bria/remove-background)
curds -no-tui -provider replicate -model remove-bg -input-image photo.jpg -output cutout.png

# Upscale / super-resolution (nightmareai/real-esrgan)
curds -no-tui -provider replicate -model upscale -input-image small.jpg -scale 4 -output big.png
```

- `-scale`: upscale factor, `1`–`10` (default `4`). Upscale only.
- `-face-enhance`: run GFPGAN face restoration; helps on portraits and
  low-resolution faces. Upscale only.

## Variants

`-number-of-images N` generates N variants in one call (OpenAI images).

## Auth resolution order

1. `-token`
2. `~/.config/curds/config.toml` or `$CURDS_CONFIG`
3. `.env` in the current directory
4. `OPENAI_API_KEY` (OpenAI) / `REPLICATE_API_TOKEN` (Replicate)

If both provider tokens are present and `-provider` is omitted, curds prefers
OpenAI. `curds` has no `--version` flag; the startup log printed by
`curds --help` includes the version.
