# Curds CLI — full parameter reference

Written from `curds --help` for version `0.6.0`; if the installed help text
differs, prefer the current `curds --help` output.

## Output formats

`-output-format` accepts `webp`, `png`, `jpeg`, `mp4`, `mp3`, or `wav`; the
output file extension also drives the format unless the flag is set. For
ordinary video generation, `.mp4` uses the current default video model unless
`-provider` or `-model` overrides it; `.mp3`/`.wav` require an audio model
(`-model music`, `music-vocal`, `sfx`, or the text-to-speech models). OpenAI
supports `-output-compression` for WebP and JPEG.

## Image sizes and aspect ratios

OpenAI `gpt-image-2.5` / `gpt-image-2` named aspect ratios: `1:1`, `3:2`, `2:3`, `4:3`, `3:4`,
`16:9`, `9:16`, `21:9`, `9:21`, `2:1`, `1:2`, `16:9-4k`, `9:16-4k`.
Custom `-size WxH` is OpenAI-only and is rounded to model constraints.
Replicate's `openai/gpt-image-2` wrapper only accepts `1:1`, `3:2`, and `2:3`.

Transparent image generation is not supported by `gpt-image-2.5` / `gpt-image-2`; use
`-model remove-bg` for transparent PNG cutouts from an existing image.

## Reference images and masks

- `-input-image` accepts file paths, `http(s)` URLs, or `data:` URLs.
- Repeat `-input-image` or pass comma-separated values, up to 16 total for
  OpenAI edits.
- `-mask` is OpenAI edits only and must be a PNG with an alpha channel:

```bash
curds -no-tui -provider openai -input-image source.png -mask mask.png -prompt "$PROMPT" -output "$OUT"
```

## Native xAI video

`-provider xai` (with or without `-model`) selects native `grok-imagine-video`: `480p` or `720p`
(default `720p`), 1–15 seconds. It does not support 1080p. Curds rejects
unsupported resolution before submitting a job; no automatic downgrade or
model switch occurs. For 1080p, explicitly select a compatible model such as
`-provider replicate -model seedance-2`, considering cost and input needs.
Do not confuse this adapter with native `grok-imagine-video-1.5`.
For a small video inset, generate at its intended display size and preserve
aspect ratio; do not upscale a 720p clip to claim native full-screen detail.

## Seedance 2.5 video (default mp4 model)

Default Replicate video model: `bytedance/seedance-2.5` (model key
`seedance-2.5`). Text-to-video, first/last frame, or reference-guided.

- `-video-duration`: `-1` (intelligent) or `4`–`30` seconds (default `5`).
- `-video-resolution`: `480p` or `720p` (default `720p`). No 1080p on 2.5;
  `-model seedance-2` and `-model kling-v3` have it.
- `-aspect-ratio`: `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `21:9`, `adaptive`
  (default `16:9`; no `9:21` on 2.5).
- `-input-image`: one entry is the first frame; several become references.
- On request only: `-last-frame-image`, `-reference-image` (≤30),
  `-reference-video` (≤10), `-reference-audio` (≤10), `-no-audio`, `-seed`.
- Audio is stripped automatically. Do not switch models for ordinary mp4
  generation.

### Seedance 2.0 (2.5's predecessor, for 1080p)

Model key `seedance-2` (`bytedance/seedance-2.0`): `-video-duration` `-1` or
`4`–`15`, `-video-resolution` `480p`/`720p`/`1080p`, ratio enum plus `9:21`,
references `≤9` images / `≤3` videos / `≤3` audios.

## Kling 3.0 video (only when the user asks for it)

Model key `kling-v3` (`kwaivgi/kling-v3-video`). Text-to-video or
image-to-video with start/end frames, native audio with lip-synced dialogue.

- `-video-duration`: `3`–`15` seconds (default `5`).
- `-video-resolution`: `720p`, `1080p`, `4k` (default `1080p`) — maps to
  Kling's standard/pro/4k mode.
- `-aspect-ratio`: `16:9`, `9:16`, `1:1` (default `16:9`).
- `-input-image`: 0 or 1 start frame; `-last-frame-image` needs one.
- No reference images/videos/audio and no `-seed`; use `seedance-2.5` for those.

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

## Alternative image models (only when the user asks, or for cost)

`gpt-image-2.5` via `-provider openai` is the default image model (OpenAI id
`gpt-image-2.5-flare`, fast everyday 2.5). `-model gpt-image-2.5-sunburst`
(alias `sunburst`) is GPT Image 2.5 Sunburst, the larger 2.5 model.
`-model gpt-image-2` picks the previous generation.
They take the same size, quality (`low` / `medium` / `high` / `xhigh` /
`max` / `auto`), background, moderation, and output-format
parameters as `gpt-image-2`. Two Replicate models
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

## Talking heads and lip-sync (Replicate, only when the user asks)

Both are media-in/video-out; there is no prompt requirement and their audio is
never stripped by default (`-strip-audio` defaults to false for these two).

`-model kling-avatar` (`kwaivgi/kling-avatar-v2`): one portrait plus one audio
clip become a lip-synced talking head (~5 min).

- `-input-image`: exactly one portrait (jpg/png, ≤10MB).
- `-audio`: mp3, wav, m4a, or aac, ≤5MB.
- `-prompt`: optional (actions/emotion/camera).
- `-video-resolution`: `720p` (std) or `1080p` (pro, default).

`-model lipsync` (`sync/lipsync-2-pro`): re-animates the mouth in an existing
video to match new audio.

- `-input-video`: mp4. `-audio`: wav.
- `-sync-mode`: `loop` (default), `bounce`, `cut_off`, `silence`, `remap`.
- `-sync-temperature`: 0–1 (default 0.5). `-active-speaker`: bool.
- No `-prompt`; passing one is rejected locally.

`-crop-captions` crops a generated video to the top 74% of the frame
(centered) to remove a burned-in caption band; ffmpeg re-encodes at crf 16 and
copies audio, and the flag is a no-op without ffmpeg.

## Music and sound effects (Replicate, only when the user asks)

Prompt-in/audio-out; no input media, no `-aspect-ratio` / `-size` /
`-quality` (not sent). Output is `mp3` unless `-output` ends in `.wav`.
`curds run OWNER/MODEL key=value …` drives any other audio model.

`-model music` (`elevenlabs/music`, the default music model): score, cue, or
loop. Instrumental by default (`-instrumental=false` for vocals); honors
`-duration` exactly, 5–300s (default 10), sent as `music_length_ms`. No
`-lyrics`, no `-seed`.

```bash
curds -no-tui -model music -duration 45 -prompt "$PROMPT" -output "$OUT.mp3"
```

`-model music-vocal` (`minimax/music-2.6`, alias `minimax-music`): a full song
with vocals. `-prompt` sets style; `-lyrics TEXT` or `-lyrics @file.txt` gives
the words (`[Verse]`/`[Chorus]` tags and newlines pass through), and lyrics
work with no prompt. Without lyrics the model writes them from the prompt.
It ignores length upstream (2–3 min renders), so with `-duration` curds trims
the download with a 2s fade-out via ffmpeg (`event=audio.trimmed`; no ffmpeg →
`audio.trim_skipped`, full render kept). No `-seed`.

```bash
curds -no-tui -model music-vocal -lyrics @song.txt -duration 60 -prompt "$PROMPT" -output "$OUT.mp3"
```

`-model sfx` (`stability-ai/stable-audio-2.5`): sound effects, ambience, short
cues. `-duration` 1–190s (default 10), optional `-seed`. It picks its own
container: curds transcodes to the requested one via ffmpeg when available,
else saves the returned container under its own extension
(`event=audio.format_mismatch`).

```bash
curds -no-tui -model sfx -duration 8 -prompt "$PROMPT" -output "$OUT.wav"
```

`-lyrics` anywhere but `music-vocal` is a usage error (exit 2), as is a
duration outside the model's range or an audio output format without an audio
model.

## Text to speech (only when the user asks)

Prompt-in/audio-out (text from `-prompt` or stdin); no input media, no
`-aspect-ratio` / `-size` / `-quality`. Output is `mp3` unless `-output` ends
in `.wav`. `tts` / `tts-minimax` / `tts-elevenlabs` need a Replicate token;
`tts-openai` / `tts-1-hd` use the `openai` provider's `/v1/audio/speech`.

`-model tts` (`google/gemini-3.1-flash-tts`, the default TTS model): 30
voices, 70+ languages (the language follows the model's own default,
`en-US`), and a natural-language style prompt. Text and style prompt are each
capped at 4000 bytes. `-voice` is one of Achernar, Achird, Algenib, Algieba,
Alnilam, Aoede, Autonoe, Callirrhoe, Charon, Despina, Enceladus, Erinome,
Fenrir, Gacrux, Iapetus, Kore, Laomedeia, Leda, Orus, Pulcherrima, Puck,
Rasalgethi, Sadachbia, Sadaltager, Schedar, Sulafat, Umbriel, Vindemiatrix,
Zephyr, Zubenelgenubi (default Kore).
`-instructions` is the style prompt ("warm and slow, British accent"). No
`-speed`, no `-emotion` / `-pitch`, no `-stability` / `-style`. The model
returns WAV; a `.mp3` output is transcoded locally.

```bash
curds -no-tui -model tts -voice Kore -instructions "warm and slow, British accent" -prompt "$PROMPT" -output "$OUT.wav"
```

`-model tts-minimax` (`minimax/speech-2.8-hd`): narration up to 10000
characters (supports `<#0.5#>` pause markers). `-voice` is a free-form voice
id (system or cloned; default `English_Wiselady`);
`-emotion` `auto|happy|sad|angry|fearful|disgusted|surprised|calm|fluent|
neutral` (default `auto`); `-speed` 0.5–2; `-pitch` -12..12. curds asks for
44.1 kHz output (256 kbps mp3) and English normalization.

```bash
curds -no-tui -model tts-minimax -voice English_Deep-VoicedGentleman -emotion calm -prompt "$PROMPT" -output "$OUT.mp3"
```

`-model tts-elevenlabs` (`elevenlabs/v3`): expressive TTS; the text is read
verbatim, so bracketed emotion markers in it are spoken aloud rather than
interpreted (the Replicate deployment does not parse them). `-voice` is one of
Rachel, Drew, Clyde, Paul, Aria, Domi, Dave, Roger, Fin, Sarah, James, Jane,
Juniper, Arabella, Hope, Bradford, Reginald, Gaming, Austin, Kuon, Blondie,
Priyanka, Alexandra, Monika, Mark, Grimblewood (default Rachel). `-speed`
0.7–1.2, `-stability` 0–1 (default 0.5), `-style` 0–1 (default 0). Returns
mp3; `.wav` is transcoded locally.

```bash
curds -no-tui -model tts-elevenlabs -voice Rachel -style 0.8 -prompt "$PROMPT" -output "$OUT.mp3"
```

`-model tts-openai` (`gpt-4o-mini-tts`, the OpenAI speech endpoint) and
`-model tts-1-hd` (the earlier model, no `-instructions`). Text is capped at
4096 characters. `-voice` is one of alloy, ash, ballad, coral, echo, fable,
onyx, nova, sage, shimmer, verse, marin, cedar (default sage); `-speed`
0.25–4. `-instructions` (accent/tone, e.g. `"crisp British RP, dry"`) is
`tts-openai` only.

```bash
curds -no-tui -model tts-openai -voice sage -instructions "crisp British RP, dry" -prompt "$PROMPT" -output "$OUT.wav"
cat script.txt | curds -no-tui -model tts-minimax -output "$OUT.wav"
```

`-emotion` / `-pitch` apply to `tts-minimax` only, `-stability` / `-style` to
`tts-elevenlabs` only, and `-instructions` to `tts` and `tts-openai`; `-speed`
applies to `tts-minimax` / `tts-elevenlabs` / `tts-openai` (Gemini has no
speed knob); anything else is a usage error (exit 2). List MiniMax voices with
`curds run -schema minimax/speech-2.8-hd` and Gemini voices with
`curds run -schema google/gemini-3.1-flash-tts`.

## Raw Replicate passthrough (`curds run`)

For models curds does not wrap (or when you need exact inputs):

```bash
curds run -schema OWNER/MODEL                       # list inputs, then exit
curds run OWNER/MODEL prompt="text" duration=5      # run, download output
curds run -json OWNER/MODEL image=@photo.png        # print prediction JSON
```

- Values are JSON when they parse (`5`, `true`, `[1,2]`, `{"a":1}`, `"text"`),
  otherwise plain strings; `key=@path` uploads a local file as a data URL.
- Repeat a key for an array (`ref=@a.png ref=@b.png`); commas in one value are
  not split.
- Flags: `-output PATH`, `-schema`, `-json`, `-token`, `-poll-interval`,
  `-timeout`, `-verbose`. Multiple outputs become `PATH-1`, `PATH-2`, …;
  saved paths print to stdout.

## Seedance face rejection → Kling fallback

A Seedance prediction (2.5 or 2.0) that fails with `flagged as sensitive
(E005)` (a realistic human face in an input image) is retried once on Kling
3.0 with the first image as `start_image`, the last frame as `end_image`,
duration clamped to 3–15s, and a mapped ratio/resolution. `-no-fallback` turns
that off; the log then carries `hint="retry with -model kling-v3"`.

## Background removal and upscaling (Replicate)

Both take exactly one `-input-image` (file path, `http(s)` URL, or `data:`
URL) and no `-prompt`; `-aspect-ratio` and `-size` are ignored.

`-model remove-bg` (`bria/remove-background`) returns a transparent PNG.

`-model upscale` (`prunaai/p-image-upscale`, the default upscaler) scales each
side by `-scale` `1`–`8` (default `4`) and returns PNG by default
(`-output-format png|jpeg|webp`). It has **no** face enhancement:
`-face-enhance` is a usage error (exit 2).

`-model upscale-esrgan` (`nightmareai/real-esrgan`) is the 2021-era upscaler
with `-scale` `1`–`10` (default `4`) and GFPGAN `-face-enhance`.

`-model upscale-pro` (`topazlabs/image-upscale`) is better on faces and text:
`-scale` `2`/`4`/`6` (omit for enhance-only), `-face-enhance` requires a
`-scale`.

```bash
# Background removal → transparent PNG (bria/remove-background)
curds -no-tui -provider replicate -model remove-bg -input-image photo.jpg -output cutout.png

# Upscale 4x (prunaai/p-image-upscale)
curds -no-tui -provider replicate -model upscale -input-image small.jpg -scale 4 -output big.png

# Portrait with face enhancement (nightmareai/real-esrgan)
curds -no-tui -provider replicate -model upscale-esrgan -face-enhance -input-image headshot.jpg -output headshot-4x.png
```

- `-scale`: per-side upscale factor, model-specific range (see above).
- `-face-enhance`: GFPGAN face restoration; `upscale-esrgan` / `upscale-pro`
  only.

## Variants

`-number-of-images N` generates N variants in one call (OpenAI images).

## Auth resolution order

1. `-token`
2. `~/.config/curds/config.toml` or `$CURDS_CONFIG`
3. `.env` in the current directory
4. `OPENAI_API_KEY` (OpenAI) / `REPLICATE_API_TOKEN` (Replicate)

Text-to-speech and music/sound-effect models bind to the provider that runs
them (`tts`/`tts-minimax`/`tts-elevenlabs`/`music`/`sfx` → Replicate,
`tts-openai`/`tts-1-hd` → OpenAI), so an explicit incompatible
`-provider`/`-model` pair is rejected locally.

If both provider tokens are present and `-provider` is omitted, curds prefers
OpenAI. `curds` has no `--version` flag; the startup log printed by
`curds --help` includes the version.
