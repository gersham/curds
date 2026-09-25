package curds

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"time"
)

// ReplicateProvider talks to the Replicate REST API.
//
// For an official model like "owner/name" it uses the dedicated
// /v1/models/{owner}/{name}/predictions endpoint. For a versioned
// "owner/name:version" reference it uses /v1/predictions.
type ReplicateProvider struct {
	HTTPClient *http.Client
	APIBase    string // default https://api.replicate.com/v1
}

func (p *ReplicateProvider) Name() string { return ProviderReplicate }

func (p *ReplicateProvider) base() string {
	if p.APIBase != "" {
		return p.APIBase
	}
	return "https://api.replicate.com/v1"
}

type replicatePrediction struct {
	ID     string            `json:"id"`
	Status string            `json:"status"`
	Output json.RawMessage   `json:"output"`
	Error  any               `json:"error"`
	Logs   string            `json:"logs"`
	URLs   map[string]string `json:"urls"`

	// raw is the last upstream JSON body for this prediction, kept verbatim so
	// `curds run -json` can print what Replicate actually returned.
	raw json.RawMessage
}

func (p *ReplicateProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	logInfo(req, "generation.started",
		"provider", "replicate",
		"model", req.Model,
		"prompt_chars", len(req.Prompt),
		"num_images", req.NumImages,
		"aspect_ratio", req.AspectRatio,
	)

	input, err := buildReplicateInput(req)
	if err != nil {
		logError(req, "replicate.input_media_failed", "err", err.Error())
		return nil, err
	}

	pred, err := p.runPrediction(ctx, req, input)
	if err != nil {
		return p.fallbackOnSeedanceFaceRejection(ctx, req, err)
	}
	return p.resultFromPrediction(ctx, req, pred)
}

// runPrediction creates one prediction and polls it to a terminal status,
// returning it when it succeeded. A non-succeeded status comes back as an
// error carrying the upstream failure text.
func (p *ReplicateProvider) runPrediction(ctx context.Context, req *Request, input map[string]any) (*replicatePrediction, error) {
	pred, err := p.createPrediction(ctx, req, input)
	if err != nil {
		return nil, err
	}
	logInfo(req, "replicate.created", "id", pred.ID, "status", pred.Status)

	pred, err = p.waitForPrediction(ctx, req, pred)
	if err != nil {
		return nil, err
	}
	if pred.Status != "succeeded" {
		logError(req, "replicate.failed", "id", pred.ID, "status", pred.Status, "msg", formatErr(pred.Error))
		return nil, fmt.Errorf("prediction %s: %s", pred.Status, formatErr(pred.Error))
	}
	return pred, nil
}

// resultFromPrediction downloads a succeeded prediction's output, picking the
// video or image path based on the model.
func (p *ReplicateProvider) resultFromPrediction(ctx context.Context, req *Request, pred *replicatePrediction) (*Result, error) {
	urls, err := extractOutputURLs(pred.Output)
	if err != nil {
		return nil, fmt.Errorf("parse output: %w", err)
	}
	if len(urls) == 0 {
		return nil, errors.New("prediction succeeded but produced no output URLs")
	}
	if IsAudioModel(req.Model) {
		return p.downloadAudios(ctx, req, pred.ID, urls)
	}
	if IsVideoModel(req.Model) {
		return p.downloadVideos(ctx, req, pred.ID, urls)
	}
	return p.downloadImages(ctx, req, pred.ID, urls)
}

// fallbackOnSeedanceFaceRejection retries a face-rejected Seedance prediction
// once on Kling 3.0. Seedance refuses any input image containing a realistic
// human face ("flagged as sensitive", E005) while Kling accepts the same
// frames, so the fallback turns a hard failure into a render. Anything that is
// not a face rejection, a request with no image input, or -no-fallback comes
// back as the original error.
func (p *ReplicateProvider) fallbackOnSeedanceFaceRejection(ctx context.Context, req *Request, predErr error) (*Result, error) {
	if !IsSeedanceModel(req.Model) || !hasSeedanceImageInput(req) || !seedanceFaceRejected(predErr) {
		return nil, predErr
	}
	if req.NoFallback {
		logInfo(req, "model.fallback_disabled",
			"from", req.Model, "reason", "seedance_rejected_face", "hint", "retry with -model kling-v3")
		return nil, predErr
	}
	freq := klingFallbackRequest(req)
	input, err := buildReplicateKlingVideoInput(freq)
	if err != nil {
		logError(req, "model.fallback_failed", "to", freq.Model, "err", err.Error())
		return nil, predErr
	}
	logInfo(req, "model.fallback", "from", req.Model, "to", freq.Model, "reason", "seedance_rejected_face")
	pred, err := p.runPrediction(ctx, freq, input)
	if err != nil {
		logError(req, "model.fallback_failed", "to", freq.Model, "err", err.Error())
		return nil, predErr
	}
	return p.resultFromPrediction(ctx, freq, pred)
}

// seedanceFaceRejected reports whether a Seedance failure is Replicate's
// content filter refusing a realistic human face (prediction error text
// "flagged as sensitive (E005)").
func seedanceFaceRejected(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "e005") || strings.Contains(msg, "flagged as sensitive")
}

// hasSeedanceImageInput reports whether the request fed Seedance an image —
// the inputs that trip the face filter.
func hasSeedanceImageInput(r *Request) bool {
	return len(r.InputImages) > 0 || r.LastFrameImage != "" || len(r.ReferenceImages) > 0
}

// klingFallbackRequest derives the Kling 3.0 request used for the fallback.
// Only what Kling understands carries over: the prompt, the first image as
// start_image, the last frame as end_image, a duration clamped into Kling's
// 3-15s range, and a mapped resolution and aspect ratio.
func klingFallbackRequest(req *Request) *Request {
	out := *req
	out.Model = KlingVideoModel
	out.VideoResolution = klingFallbackResolution(req.VideoResolution)
	out.AspectRatio = klingFallbackAspectRatio(req.AspectRatio)
	out.VideoDuration = klingFallbackDuration(req.VideoDuration)
	out.LastFrameImage = req.LastFrameImage
	out.Seed = 0 // Kling has no seed
	out.ReferenceImages = nil
	out.ReferenceVideos = nil
	out.ReferenceAudios = nil
	out.InputImages = nil
	if len(req.InputImages) > 0 {
		out.InputImages = req.InputImages[:1]
	}
	return &out
}

// klingFallbackResolution maps Seedance's resolution vocabulary onto Kling's:
// 480p and 720p become standard (720p), 1080p becomes pro, 4k stays 4k.
func klingFallbackResolution(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "480p", "720p":
		return "720p"
	case "4k":
		return "4k"
	}
	// Seedance 1080p, or unset: Kling's own default mode is pro.
	return "1080p"
}

// klingFallbackAspectRatio passes the ratio through when Kling supports it and
// falls back to 16:9 (Kling's landscape default) otherwise.
func klingFallbackAspectRatio(ratio string) string {
	switch strings.TrimSpace(strings.ToLower(ratio)) {
	case "16:9", "9:16", "1:1":
		return ratio
	}
	return "16:9"
}

// klingFallbackDuration clamps a Seedance duration into Kling's 3-15s range.
func klingFallbackDuration(seconds int) int {
	switch {
	case seconds == 0 || seconds == -1:
		return 5 // Seedance's "intelligent"/default duration.
	case seconds < 3:
		return 3
	case seconds > 15:
		return 15
	}
	return seconds
}

func buildReplicateInput(req *Request) (map[string]any, error) {
	if IsVideoModel(req.Model) {
		return buildReplicateVideoInput(req)
	}
	if IsAudioModel(req.Model) {
		return buildReplicateAudioInput(req)
	}
	if IsSegmentationModel(req.Model) {
		return buildReplicateSegmentationInput(req)
	}
	if IsUpscaleModel(req.Model) {
		return buildReplicateUpscaleInput(req)
	}
	if IsFluxImageModel(req.Model) {
		return buildReplicateFluxInput(req)
	}
	if IsNanoBananaImageModel(req.Model) {
		return buildReplicateNanoBananaInput(req)
	}
	input := map[string]any{
		"prompt":           req.Prompt,
		"aspect_ratio":     req.AspectRatio,
		"quality":          req.Quality,
		"number_of_images": req.NumImages,
		"output_format":    req.OutputFormat,
		"background":       req.Background,
		"moderation":       req.Moderation,
	}
	if req.ReplicateBYOKey != "" {
		input["openai_api_key"] = req.ReplicateBYOKey
	}
	if len(req.InputImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["input_images"] = urls
	}
	return input, nil
}

// buildReplicateFluxInput builds the input for black-forest-labs/flux-2-pro.
// FLUX sizes output either by megapixel target (`resolution`) or by explicit
// `width`/`height` with aspect_ratio "custom". It takes reference images in
// `input_images` and has no quality/background/moderation knobs.
func buildReplicateFluxInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":        req.Prompt,
		"output_format": replicateImageFormat(req.OutputFormat),
	}
	if w, h, ok := ParseSize(req.Size); ok {
		input["aspect_ratio"] = "custom"
		input["width"] = w
		input["height"] = h
	} else {
		input["aspect_ratio"] = req.AspectRatio
	}
	if res := FluxImageResolution(req.ImageResolution); res != "" {
		input["resolution"] = res
	}
	if req.OutputCompression > 0 {
		input["output_quality"] = req.OutputCompression
	}
	if req.Seed != 0 {
		input["seed"] = req.Seed
	}
	if len(req.InputImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["input_images"] = urls
	}
	return input, nil
}

// buildReplicateNanoBananaInput builds the input for google/nano-banana-2. It
// names its reference array `image_input`, sizes output with 1K/2K/4K, and
// emits jpg or png only.
func buildReplicateNanoBananaInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":        req.Prompt,
		"aspect_ratio":  req.AspectRatio,
		"output_format": replicateImageFormat(req.OutputFormat),
	}
	if res := NanoBananaImageResolution(req.ImageResolution); res != "" {
		input["resolution"] = res
	}
	if len(req.InputImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["image_input"] = urls
	}
	return input, nil
}

// replicateImageFormat maps curds' output format onto the spelling used by
// FLUX.2 and Nano Banana 2, which say "jpg" where curds says "jpeg".
func replicateImageFormat(format string) string {
	if format == "jpeg" {
		return "jpg"
	}
	return format
}

// buildReplicateSegmentationInput builds the input for bria/remove-background
// (and any future SAM-style models we wire in). The model takes a single
// `image` URL/data-URL and returns one transparent PNG. No prompt, no aspect
// ratio, no quality knob.
func buildReplicateSegmentationInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare segmentation input image: %w", err)
	}
	return map[string]any{"image": urls[0]}, nil
}

// buildReplicateUpscaleInput builds the input for nightmareai/real-esrgan
// (and any future Real-ESRGAN-style super-resolution model we wire in). The
// model takes a single `image` URL/data-URL, a numeric `scale` factor, and an
// optional `face_enhance` flag, and returns one upscaled image. No prompt, no
// aspect ratio, no quality knob.
func buildReplicateUpscaleInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare upscale input image: %w", err)
	}
	if IsTopazUpscaleModel(req.Model) {
		input := map[string]any{
			"image":          urls[0],
			"upscale_factor": TopazUpscaleFactor(req.Scale),
			"output_format":  "png",
		}
		if req.FaceEnhance {
			input["face_enhancement"] = true
		}
		return input, nil
	}
	scale := req.Scale
	if scale == 0 {
		scale = DefaultUpscaleScale
	}
	input := map[string]any{
		"image": urls[0],
		"scale": scale,
	}
	if req.FaceEnhance {
		input["face_enhance"] = true
	}
	return input, nil
}

// buildReplicateAudioInput builds the input for whichever audio model curds
// wraps: ElevenLabs Music (score), MiniMax Music 2.6 (song), or Stable Audio
// 2.5 (sound effects). Each names its length and container differently, so the
// per-model builders below do the mapping.
func buildReplicateAudioInput(req *Request) (map[string]any, error) {
	switch {
	case IsTTSSpeechModel(req.Model):
		return buildReplicateSpeechInput(req), nil
	case IsTTSElevenLabsModel(req.Model):
		return buildReplicateElevenLabsTTSInput(req), nil
	case IsMusicModel(req.Model):
		return buildReplicateMusicInput(req), nil
	case IsMusicVocalModel(req.Model):
		return buildReplicateMusicVocalInput(req), nil
	case IsSFXModel(req.Model):
		return buildReplicateSFXInput(req), nil
	}
	return nil, fmt.Errorf("unsupported audio model %q", req.Model)
}

// elevenLabsMusicFormat maps curds' mp3/wav onto elevenlabs/music's
// output_format enum. The high-fidelity entry wins for each container: MP3
// high quality, or CD-quality WAV (44.1 kHz, 16-bit).
func elevenLabsMusicFormat(format string) string {
	if format == "wav" {
		return "wav_cd_quality"
	}
	return "mp3_high_quality"
}

// instrumentalValue reads Request.Instrumental, falling back to the model's
// own default when the caller left it unset (applyDefaults normally fills it).
func instrumentalValue(req *Request, fallback bool) bool {
	if req.Instrumental != nil {
		return *req.Instrumental
	}
	return fallback
}

// buildReplicateMusicInput builds the input for elevenlabs/music: the prompt,
// an exact music_length_ms (5-300s), force_instrumental (default true), and
// the output_format enum. The model honors the requested length exactly.
func buildReplicateMusicInput(req *Request) map[string]any {
	input := map[string]any{
		"prompt":             req.Prompt,
		"force_instrumental": instrumentalValue(req, true),
		"output_format":      elevenLabsMusicFormat(req.OutputFormat),
	}
	if req.Duration > 0 {
		input["music_length_ms"] = int(math.Round(req.Duration * 1000))
	}
	return input
}

// buildReplicateMusicVocalInput builds the input for minimax/music-2.6: the
// prompt, is_instrumental (default false — vocals), lyrics when supplied, and
// CD-grade mp3/wav output (44.1 kHz, 256 kbps). With no lyrics on a vocal
// track the model's own lyrics_optimizer writes them from the prompt. The
// requested length is NOT sent: the model renders 2-3 minutes and curds trims
// locally instead (see the CLI's -duration handling).
func buildReplicateMusicVocalInput(req *Request) map[string]any {
	instrumental := instrumentalValue(req, false)
	input := map[string]any{
		"prompt":          req.Prompt,
		"is_instrumental": instrumental,
		"audio_format":    req.OutputFormat,
		"sample_rate":     44100,
		"bitrate":         256000,
	}
	if strings.TrimSpace(req.Lyrics) != "" {
		input["lyrics"] = req.Lyrics
	} else if !instrumental {
		input["lyrics_optimizer"] = true
	}
	return input
}

// buildReplicateSpeechInput builds the input for minimax/speech-2.8-hd: the
// text, a voice id (free-form — system voices and cloned ids both work), the
// delivery knobs it accepts, and CD-grade output (44.1 kHz; 256 kbps for mp3,
// which WAV does not carry). English normalization stays on so numbers and
// dates read naturally.
func buildReplicateSpeechInput(req *Request) map[string]any {
	input := map[string]any{
		"text":                  req.Prompt,
		"voice_id":              req.Voice,
		"audio_format":          req.OutputFormat,
		"sample_rate":           44100,
		"english_normalization": true,
	}
	if req.OutputFormat == "mp3" {
		input["bitrate"] = 256000
	}
	if req.Emotion != "" {
		input["emotion"] = req.Emotion
	}
	if req.Speed > 0 {
		input["speed"] = req.Speed
	}
	if req.Pitch != 0 {
		input["pitch"] = req.Pitch
	}
	return input
}

// buildReplicateElevenLabsTTSInput builds the input for elevenlabs/v3: the
// text goes in `prompt`, plus the voice and the optional stability/style/speed
// knobs. The model returns mp3; a wav request is transcoded after download,
// like stable-audio-2.5's container mismatch.
func buildReplicateElevenLabsTTSInput(req *Request) map[string]any {
	input := map[string]any{
		"prompt": req.Prompt,
		"voice":  req.Voice,
	}
	if req.Speed > 0 {
		input["speed"] = req.Speed
	}
	if req.Stability != nil {
		input["stability"] = *req.Stability
	}
	if req.Style != nil {
		input["style"] = *req.Style
	}
	return input
}

// buildReplicateSFXInput builds the input for stability-ai/stable-audio-2.5:
// the prompt, an integer duration in seconds, and an optional seed. It ignores
// the container request — the model returns what it returns — so the CLI
// reconciles the extension after download.
func buildReplicateSFXInput(req *Request) map[string]any {
	duration := int(math.Round(req.Duration))
	if duration <= 0 {
		duration = DefaultSFXDuration
	}
	input := map[string]any{
		"prompt":   req.Prompt,
		"duration": duration,
	}
	if req.Seed != 0 {
		input["seed"] = req.Seed
	}
	return input
}

func buildReplicateVideoInput(req *Request) (map[string]any, error) {
	switch {
	case IsGrokImagineVideoModel(req.Model):
		return buildReplicateGrokImagineVideoInput(req)
	case IsMinimaxVideoModel(req.Model):
		return buildReplicateMinimaxVideoInput(req)
	case IsKlingVideoModel(req.Model):
		return buildReplicateKlingVideoInput(req)
	case IsKlingAvatarModel(req.Model):
		return buildReplicateKlingAvatarInput(req)
	case IsLipsyncModel(req.Model):
		return buildReplicateLipsyncInput(req)
	}
	return buildReplicateSeedanceVideoInput(req)
}

// buildReplicateKlingVideoInput builds the input for kwaivgi/kling-v3-video.
// Kling names its frames `start_image` / `end_image` and expresses resolution
// as a `mode` (standard = 720p, pro = 1080p, 4k). Audio is opt-in upstream, so
// curds sends generate_audio=true unless -no-audio was passed, matching how it
// treats Seedance.
func buildReplicateKlingVideoInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":         req.Prompt,
		"mode":           KlingMode(req.VideoResolution),
		"aspect_ratio":   req.AspectRatio,
		"duration":       req.VideoDuration,
		"generate_audio": true,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	if req.GenerateAudio != nil {
		input["generate_audio"] = *req.GenerateAudio
	}
	if len(req.InputImages) == 1 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare start image: %w", err)
		}
		input["start_image"] = urls[0]
	}
	if req.LastFrameImage != "" {
		urls, err := encodeMediaAsDataURLs([]string{req.LastFrameImage})
		if err != nil {
			return nil, fmt.Errorf("prepare end image: %w", err)
		}
		input["end_image"] = urls[0]
	}
	return input, nil
}

// buildReplicateKlingAvatarInput builds the input for kwaivgi/kling-avatar-v2:
// one portrait `image` plus one `audio` clip, with an optional prompt for
// actions/emotion/camera. Resolution is expressed as the model's `mode` enum
// via -video-resolution (720p = std, 1080p = pro).
func buildReplicateKlingAvatarInput(req *Request) (map[string]any, error) {
	images, err := encodeMediaAsDataURLs(req.InputImages[:1])
	if err != nil {
		return nil, fmt.Errorf("prepare portrait image: %w", err)
	}
	audios, err := encodeMediaAsDataURLs([]string{req.Audio})
	if err != nil {
		return nil, fmt.Errorf("prepare audio: %w", err)
	}
	input := map[string]any{
		"image": images[0],
		"audio": audios[0],
		"mode":  KlingAvatarMode(req.VideoResolution),
	}
	if strings.TrimSpace(req.Prompt) != "" {
		input["prompt"] = req.Prompt
	}
	return input, nil
}

// buildReplicateLipsyncInput builds the input for sync/lipsync-2-pro: a source
// `video` and an `audio` clip, plus the optional sync_mode, temperature, and
// active_speaker knobs. There is no prompt.
func buildReplicateLipsyncInput(req *Request) (map[string]any, error) {
	videos, err := encodeMediaAsDataURLs([]string{req.InputVideo})
	if err != nil {
		return nil, fmt.Errorf("prepare input video: %w", err)
	}
	audios, err := encodeMediaAsDataURLs([]string{req.Audio})
	if err != nil {
		return nil, fmt.Errorf("prepare audio: %w", err)
	}
	input := map[string]any{
		"video":          videos[0],
		"audio":          audios[0],
		"active_speaker": req.ActiveSpeaker,
	}
	if mode := strings.ToLower(strings.TrimSpace(req.SyncMode)); mode != "" {
		input["sync_mode"] = mode
	}
	if req.SyncTemperature >= 0 {
		input["temperature"] = req.SyncTemperature
	}
	return input, nil
}

// buildReplicateMinimaxVideoInput builds the input for minimax/h3. H3 names its
// fields differently from Seedance: `first_frame_image` / `last_frame_image`
// instead of `image`, `reference_*_urls` instead of `reference_*`, and `ratio`
// instead of `aspect_ratio`. It has no seed or audio toggle.
func buildReplicateMinimaxVideoInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":     req.Prompt,
		"duration":   req.VideoDuration,
		"resolution": MinimaxVideoResolution(req.VideoResolution),
		"ratio":      req.AspectRatio,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	if len(req.InputImages) == 1 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare first frame image: %w", err)
		}
		input["first_frame_image"] = urls[0]
	}
	if req.LastFrameImage != "" {
		urls, err := encodeMediaAsDataURLs([]string{req.LastFrameImage})
		if err != nil {
			return nil, fmt.Errorf("prepare last frame image: %w", err)
		}
		input["last_frame_image"] = urls[0]
	}
	if len(req.ReferenceImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		input["reference_image_urls"] = urls
	}
	if len(req.ReferenceVideos) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceVideos)
		if err != nil {
			return nil, fmt.Errorf("prepare reference videos: %w", err)
		}
		input["reference_video_urls"] = urls
	}
	if len(req.ReferenceAudios) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceAudios)
		if err != nil {
			return nil, fmt.Errorf("prepare reference audios: %w", err)
		}
		input["reference_audio_urls"] = urls
	}
	return input, nil
}

func buildReplicateGrokImagineVideoInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare input image: %w", err)
	}
	input := map[string]any{
		"prompt":       req.Prompt,
		"image":        urls[0],
		"duration":     req.VideoDuration,
		"resolution":   req.VideoResolution,
		"aspect_ratio": req.AspectRatio,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	return input, nil
}

func buildReplicateSeedanceVideoInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":         req.Prompt,
		"duration":       req.VideoDuration,
		"resolution":     req.VideoResolution,
		"aspect_ratio":   req.AspectRatio,
		"generate_audio": true,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	if req.GenerateAudio != nil {
		input["generate_audio"] = *req.GenerateAudio
	}
	if req.Seed != 0 {
		input["seed"] = req.Seed
	}
	switch {
	case len(req.InputImages) == 1:
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare first frame image: %w", err)
		}
		input["image"] = urls[0]
	case len(req.InputImages) > 1:
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		input["reference_images"] = urls
	}
	if req.LastFrameImage != "" {
		urls, err := encodeMediaAsDataURLs([]string{req.LastFrameImage})
		if err != nil {
			return nil, fmt.Errorf("prepare last frame image: %w", err)
		}
		input["last_frame_image"] = urls[0]
	}
	if len(req.ReferenceImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		if existing, ok := input["reference_images"].([]string); ok {
			input["reference_images"] = append(existing, urls...)
		} else {
			input["reference_images"] = urls
		}
	}
	if len(req.ReferenceVideos) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceVideos)
		if err != nil {
			return nil, fmt.Errorf("prepare reference videos: %w", err)
		}
		input["reference_videos"] = urls
	}
	if len(req.ReferenceAudios) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceAudios)
		if err != nil {
			return nil, fmt.Errorf("prepare reference audios: %w", err)
		}
		input["reference_audios"] = urls
	}
	return input, nil
}

// downloadAudios downloads each rendered audio file. The container is taken
// from the output URL when it names one (some models ignore the requested
// format), falling back to the requested output format.
func (p *ReplicateProvider) downloadAudios(ctx context.Context, req *Request, id string, urls []string) (*Result, error) {
	logInfo(req, "replicate.succeeded", "id", id, "audio_count", len(urls))
	res := &Result{Audios: make([]Audio, 0, len(urls))}
	for i, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			logError(req, "audio.download_failed", "index", i, "url", u, "err", err.Error())
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		format := audioFormatFromURL(u, req.OutputFormat)
		logInfo(req, "audio.downloaded", "index", i, "bytes", len(b), "format", format)
		res.Audios = append(res.Audios, Audio{Bytes: b, Format: format, URL: u})
	}
	return res, nil
}

// audioFormatFromURL reports the container an output URL names (mp3 or wav),
// falling back to the requested format for opaque URLs.
func audioFormatFromURL(rawURL, fallback string) string {
	if u, err := url.Parse(rawURL); err == nil {
		switch ext := strings.ToLower(strings.TrimPrefix(path.Ext(u.Path), ".")); ext {
		case "mp3", "wav":
			return ext
		}
	}
	return fallback
}

func (p *ReplicateProvider) downloadImages(ctx context.Context, req *Request, id string, urls []string) (*Result, error) {
	logInfo(req, "replicate.succeeded", "id", id, "image_count", len(urls))
	res := &Result{Images: make([]Image, 0, len(urls))}
	for i, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			logError(req, "image.download_failed", "index", i, "url", u, "err", err.Error())
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		logInfo(req, "image.downloaded", "index", i, "bytes", len(b), "format", req.OutputFormat)
		res.Images = append(res.Images, Image{Bytes: b, Format: req.OutputFormat})
	}
	return res, nil
}

func (p *ReplicateProvider) downloadVideos(ctx context.Context, req *Request, id string, urls []string) (*Result, error) {
	logInfo(req, "replicate.succeeded", "id", id, "video_count", len(urls))
	res := &Result{Videos: make([]Video, 0, len(urls))}
	for i, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			logError(req, "video.download_failed", "index", i, "url", u, "err", err.Error())
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		logInfo(req, "video.downloaded", "index", i, "bytes", len(b), "format", req.OutputFormat)
		res.Videos = append(res.Videos, Video{Bytes: b, Format: req.OutputFormat, URL: u})
	}
	return res, nil
}

func (p *ReplicateProvider) createPrediction(ctx context.Context, req *Request, input map[string]any) (*replicatePrediction, error) {
	body := map[string]any{"input": input}
	var endpoint string
	if idx := strings.Index(req.Model, ":"); idx >= 0 {
		body["version"] = req.Model[idx+1:]
		endpoint = p.base() + "/predictions"
	} else {
		endpoint = fmt.Sprintf("%s/models/%s/predictions", p.base(), req.Model)
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Prefer", "wait=60")

	logDebug(req, "replicate.request", "endpoint", endpoint, "body", redactRequestBody(body))
	logInfo(req, "replicate.request", "endpoint", endpoint)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		logError(req, "replicate.transport_error", "err", err.Error())
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	logInfo(req, "replicate.response", "status_code", resp.StatusCode, "bytes", len(rb))
	if resp.StatusCode >= 400 {
		logError(req, "replicate.api_error", "status_code", resp.StatusCode, "body", truncate(string(rb), 500))
		return nil, fmt.Errorf("replicate API %d: %s", resp.StatusCode, string(rb))
	}
	var pred replicatePrediction
	if err := json.Unmarshal(rb, &pred); err != nil {
		return nil, fmt.Errorf("decode prediction: %w (body=%s)", err, string(rb))
	}
	pred.raw = rb
	return &pred, nil
}

func (p *ReplicateProvider) waitForPrediction(ctx context.Context, req *Request, pred *replicatePrediction) (*replicatePrediction, error) {
	for !isTerminalStatus(pred.Status) {
		logInfo(req, "replicate.polling", "id", pred.ID, "status", pred.Status, "interval", req.PollInterval.String())
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(req.PollInterval):
		}
		next, err := p.fetchPrediction(ctx, req, pred.URLs["get"])
		if err != nil {
			logError(req, "replicate.poll_failed", "id", pred.ID, "err", err.Error())
			return nil, err
		}
		pred = next
	}
	logInfo(req, "replicate.terminal", "id", pred.ID, "status", pred.Status)
	return pred, nil
}

func (p *ReplicateProvider) fetchPrediction(ctx context.Context, req *Request, getURL string) (*replicatePrediction, error) {
	if getURL == "" {
		return nil, errors.New("missing prediction get URL")
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, getURL, nil)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("get prediction %d: %s", resp.StatusCode, string(rb))
	}
	var pred replicatePrediction
	if err := json.Unmarshal(rb, &pred); err != nil {
		return nil, err
	}
	pred.raw = rb
	return &pred, nil
}

// Download fetches a rendered asset URL. The bearer token is attached only for
// Replicate-controlled hosts, so a model's own CDN never sees it.
func (p *ReplicateProvider) Download(ctx context.Context, token, rawURL string) ([]byte, error) {
	return p.downloadBytes(ctx, token, rawURL)
}

func (p *ReplicateProvider) downloadBytes(ctx context.Context, token, rawURL string) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if token != "" && isReplicateHost(rawURL) {
		hreq.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(b))
	}
	return io.ReadAll(resp.Body)
}

// isReplicateHost reports whether the URL points at a Replicate-controlled
// host. It compares the parsed hostname instead of substring-matching the
// raw URL — substring matches let an attacker craft URLs like
// https://evil.example/?replicate.delivery that would otherwise leak the
// bearer token.
func isReplicateHost(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil || u == nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}
	switch host {
	case "api.replicate.com", "replicate.com", "replicate.delivery":
		return true
	}
	return strings.HasSuffix(host, ".replicate.com") || strings.HasSuffix(host, ".replicate.delivery")
}

// redactRequestBody returns a JSON-marshaled view of the request body with
// secret-bearing fields scrubbed for logging.
func redactRequestBody(body map[string]any) string {
	clone := make(map[string]any, len(body))
	for k, v := range body {
		clone[k] = v
	}
	if input, ok := clone["input"].(map[string]any); ok {
		safe := make(map[string]any, len(input))
		for k, v := range input {
			if k == "openai_api_key" {
				safe[k] = "[REDACTED]"
				continue
			}
			safe[k] = v
		}
		clone["input"] = safe
	}
	b, err := json.Marshal(clone)
	if err != nil {
		return "<unloggable body>"
	}
	return string(b)
}

func isTerminalStatus(s string) bool {
	switch s {
	case "succeeded", "failed", "canceled":
		return true
	}
	return false
}

func extractOutputURLs(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []string{s}, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var anyArr []any
	if err := json.Unmarshal(raw, &anyArr); err == nil {
		out := make([]string, 0, len(anyArr))
		for _, x := range anyArr {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported output shape: %s", string(raw))
}

func encodeMediaAsDataURLs(paths []string) ([]string, error) {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if strings.HasPrefix(p, "http://") || strings.HasPrefix(p, "https://") || strings.HasPrefix(p, "data:") {
			out = append(out, p)
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		mt := detectMime(p, data)
		out = append(out, fmt.Sprintf("data:%s;base64,%s", mt, base64.StdEncoding.EncodeToString(data)))
	}
	return out, nil
}

func formatErr(e any) string {
	if e == nil {
		return ""
	}
	if s, ok := e.(string); ok {
		return s
	}
	b, _ := json.Marshal(e)
	return string(b)
}
