---
name: curds
description: Use for `$curds`, explicit curds CLI requests, or when the user wants the local `curds` CLI to generate images, edit/reference images, create videos, remove image backgrounds, or upscale images through OpenAI or Replicate.
model: haiku
---

# Curds CLI

`curds` is a local media-generation CLI. This skill is mechanical: assemble the
right flags, run one command, verify the artifact, report. The caller supplies
the generation prompt — do not rewrite it, expand scope, or launch extra jobs.

## Rules

- Readiness: `command -v curds`. If missing, stop and report; do not install.
- Always pass `-no-tui` and an explicit `-output` path. Use `-open` only when
  asked.
- One image per request unless variants are asked for. Never start video jobs,
  variant batches, or high-quality batches unless requested or clearly
  appropriate.
- Never print, paste, commit, or summarize API tokens. Auth resolves from
  `~/.config/curds/config.toml` (or `$CURDS_CONFIG`), `.env`, or
  `OPENAI_API_KEY`/`REPLICATE_API_TOKEN`; only pass `-token` when the user
  explicitly provides or authorizes one, and don't inspect config/`.env` files
  without a clear auth-debugging need.
- Provider: `-provider openai` for image generation and edits (`gpt-image-2` —
  lower latency and cost, more size options, no polling). `-provider xai`
  without `-model` selects native `grok-imagine-video`. For ordinary `.mp4`
  video omit `-provider`/`-model`; curds defaults to Replicate
  `bytedance/seedance-2.0`. Use `-model` for the Replicate alternatives —
  `flux-2-pro` / `nano-banana-2` (images), `kling-v3` / `minimax-h3` (video),
  `remove-bg`, `upscale`, `upscale-pro` — when the user asks for them or
  OpenAI access is missing. Do not pair `-provider` with a model that
  provider does not run; curds rejects the pair locally.

## Commands

```bash
# Image
curds -no-tui -provider openai -aspect-ratio 16:9 -quality high -prompt "$PROMPT" -output "$OUT"

# Edit / compose with reference images (comma-separated or repeated, max 16)
curds -no-tui -provider openai -input-image ref1.png,ref2.png -prompt "$PROMPT" -output "$OUT"

# Video (default model Seedance 2.0; seed from a still with -input-image)
curds -no-tui -input-image still.webp -prompt "$PROMPT" -aspect-ratio 16:9 -video-duration 5 -video-resolution 720p -output "$OUT.mp4"

# Cheap/fast image alternative (FLUX.2 [pro]) or 4K composite (Nano Banana 2)
curds -no-tui -model flux-2-pro -aspect-ratio 16:9 -image-resolution 2mp -prompt "$PROMPT" -output "$OUT.png"
curds -no-tui -model nano-banana-2 -image-resolution 4k -input-image a.png,b.png -prompt "$PROMPT" -output "$OUT.png"

# Background removal (transparent PNG cutout)
curds -no-tui -provider replicate -model remove-bg -input-image photo.jpg -output cutout.png

# Upscale / super-resolution (PNG; -scale 1-10, default 4; add -face-enhance for portraits)
curds -no-tui -provider replicate -model upscale -input-image small.jpg -scale 4 -output big.png

# Better upscale for faces and text (Topaz; -scale 2, 4, or 6)
curds -no-tui -model upscale-pro -input-image small.jpg -scale 4 -output big.png
```

## Verify and report

```bash
test -s "$OUT" && file "$OUT" && ls -lh "$OUT"
```

Plus `sips -g pixelWidth -g pixelHeight "$OUT"` for images or `ffprobe "$OUT"`
for videos when available. Report the provider, model, sanitized command shape,
absolute output path, and verification result — never tokens. Exit code `2` =
bad flags or missing input; `1` = upstream/generation failure or cancellation.
Report the exact blocker instead of retrying interactively.

## Non-default parameters

Only when the request needs something beyond the commands above — custom
`-size`, 4K aspect ratios, masks/inpainting, output formats and compression,
Grok/Seedance duration-resolution-aspect matrices, Seedance-only flags — read
`reference.md` in this skill directory.
