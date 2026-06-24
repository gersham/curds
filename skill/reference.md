# Curds CLI — full parameter reference

Written from `curds --help` for version `0.1.0`; if the installed help text
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

## Grok video (default mp4 model)

Default Replicate video model: `xai/grok-imagine-video-1.5`.

- `-video-duration`: `1`–`15` seconds.
- `-video-resolution`: `480p` or `720p`.
- `-aspect-ratio`: `auto`, `16:9`, `4:3`, `1:1`, `9:16`, `3:4`, `3:2`, `2:3`.
- Audio is stripped automatically. Do not force Seedance for ordinary mp4
  generation.

## Seedance video (only when the user asks for it)

```bash
curds -no-tui -provider replicate -model seedance-2 -aspect-ratio 16:9 -video-duration 5 -video-resolution 720p -prompt "$PROMPT" -output "$OUT"
```

- `-video-duration`: `-1` or `4`–`15` seconds.
- `-video-resolution`: `480p`, `720p`, or `1080p`.
- Seedance-specific flags, only on request: `-no-audio`, `-last-frame-image`,
  `-reference-image`, `-reference-video`, `-reference-audio`.

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
