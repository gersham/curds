// Package curds generates images and videos via generation providers.
//
// Supported providers:
//   - openai    (direct OpenAI Image API; default model gpt-image-2.5-flare)
//   - replicate (Replicate-hosted models; default openai/gpt-image-2)
//   - xai       (native xAI video API; model grok-imagine-video)
//
// The package is transport-agnostic and intended to be reusable from a CLI,
// HTTP service, or background worker.
package curds

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	ProviderReplicate = "replicate"
	ProviderOpenAI    = "openai"
	ProviderXai       = "xai"

	DefaultReplicateModel = "openai/gpt-image-2"
	DefaultOpenAIModel    = GPTImage25Model

	// GPTImage25Model is GPT Image 2.5 Flare, the default image model.
	// Fast everyday 2.5. Same request surface as gpt-image-2 on
	// /v1/images/generations and /v1/images/edits (model, prompt, n, size,
	// quality, background, moderation, output_format, output_compression,
	// user), so nothing about sizing, aspect ratios, or output formats
	// changes with it — only the id.
	GPTImage25Model = "gpt-image-2.5-flare"

	// GPTImage25SunburstModel is GPT Image 2.5 Sunburst, the larger 2.5
	// model. Higher quality, longer generation. Same request surface as
	// Flare. Selectable via -model gpt-image-2.5-sunburst or -model sunburst.
	GPTImage25SunburstModel = "gpt-image-2.5-sunburst"

	// GPTImage2Model is the previous-generation image model, still
	// selectable via -model gpt-image-2.
	GPTImage2Model = "gpt-image-2"

	// GrokImagineVideoModel is the Replicate-hosted Grok Imagine Video 1.5
	// wrapper (image-to-video only) and the default Replicate video model.
	GrokImagineVideoModel = "xai/grok-imagine-video-1.5"

	// DefaultVideoModel is the Replicate-hosted video model used when output is
	// MP4 and no model is supplied. Seedance 2.5 leads on reference handling
	// (up to 30 images, 10 videos, 10 audios), multi-scene continuity, native
	// audio, and durations up to 30 seconds. Seedance 2.0 is still selectable
	// with -model seedance-2 and remains the only Seedance with 1080p.
	DefaultVideoModel = Seedance25VideoModel

	// MinimaxVideoModel is MiniMax H3 on Replicate: multimodal text-to-video,
	// first/last-frame image-to-video, and reference images/videos/audio.
	// Selectable via -model minimax-h3; not a default.
	MinimaxVideoModel = "minimax/h3"

	// SeedanceVideoModel is ByteDance Seedance 2.0 on Replicate: text-to-video,
	// first/last frame, reference images/videos/audio, native synchronized
	// audio, and 1080p. Selectable via -model seedance-2.
	SeedanceVideoModel = "bytedance/seedance-2.0"

	// Seedance25VideoModel is ByteDance Seedance 2.5 on Replicate and the
	// default video model: text-to-video, first/last frame, up to 30 reference
	// images / 10 reference videos / 10 reference audios, native synchronized
	// audio, and durations up to 30s — but 480p/720p only. Selectable via
	// -model seedance-2.5.
	Seedance25VideoModel = "bytedance/seedance-2.5"

	// KlingVideoModel is Kling Video 3.0 on Replicate: text-to-video and
	// image-to-video with start/end frames, native audio with lip-synced
	// dialogue, and standard (720p) / pro (1080p) / 4k modes.
	KlingVideoModel = "kwaivgi/kling-v3-video"

	// KlingAvatarModel is Kling Avatar 2.0 on Replicate: a talking-head model
	// that animates one portrait with one audio clip. The prompt is optional
	// (actions/emotion/camera) and resolution maps onto its mode enum
	// (720p = std, 1080p = pro). Selectable via -model kling-avatar.
	KlingAvatarModel = "kwaivgi/kling-avatar-v2"

	// LipsyncModel is Sync Labs' lipsync-2-pro on Replicate: it re-animates a
	// mouth in an existing video to match an audio clip. No prompt; the source
	// video and audio are the whole input. Selectable via -model lipsync.
	LipsyncModel = "sync/lipsync-2-pro"

	// MusicModel is ElevenLabs Music on Replicate and the default music model:
	// a score or loop from a prompt, instrumental by default, in mp3 or wav,
	// honoring the requested length exactly. Selectable via -model music.
	MusicModel = "elevenlabs/music"

	// MusicVocalModel is MiniMax Music 2.6 on Replicate: a full song (vocals or
	// instrumental) from a prompt, optionally with caller-supplied -lyrics and
	// [Verse]/[Chorus] tags. It ignores the requested length upstream (renders
	// are 2-3 minutes) so curds trims the result locally when -duration is set.
	// Selectable via -model music-vocal (alias -model minimax-music).
	MusicVocalModel = "minimax/music-2.6"

	// SFXModel is Stable Audio 2.5 on Replicate: sound effects, ambience, and
	// short music cues from a prompt, 1-190 seconds, with an optional seed.
	// Selectable via -model sfx.
	SFXModel = "stability-ai/stable-audio-2.5"

	// DefaultSFXDuration is the length curds asks Stable Audio for when
	// -duration is omitted, in seconds.
	DefaultSFXDuration = 10

	// TTSGeminiModel is Google's Gemini 3.1 Flash TTS on Replicate and the
	// default text-to-speech model: 30 voices, 70+ languages, and a
	// natural-language style prompt (curds' -instructions) steering tone,
	// pace, and accent. It returns WAV, so an mp3 request is transcoded.
	// Selectable via -model tts.
	TTSGeminiModel = "google/gemini-3.1-flash-tts"

	// TTSSpeechModel is MiniMax Speech 2.8 HD on Replicate: natural narration
	// from text, with a free-form voice id, an emotion, speed, pitch, and an
	// mp3/wav container. Selectable via -model tts-minimax.
	TTSSpeechModel = "minimax/speech-2.8-hd"

	// TTSElevenLabsModel is ElevenLabs v3 on Replicate: expressive
	// text-to-speech with per-voice stability/style. Selectable via
	// -model tts-elevenlabs.
	TTSElevenLabsModel = "elevenlabs/v3"

	// TTSOpenAIModel is OpenAI's gpt-4o-mini-tts speech model, served by the
	// openai provider's POST /v1/audio/speech endpoint. Like Gemini 3.1 Flash
	// TTS it accepts delivery instructions ("crisp British RP, dry").
	// Selectable via -model tts-openai.
	TTSOpenAIModel = "gpt-4o-mini-tts"

	// TTS1HDModel is OpenAI's earlier tts-1-hd speech model, also served by
	// POST /v1/audio/speech. It has no -instructions knob. Selectable via
	// -model tts-1-hd.
	TTS1HDModel = "tts-1-hd"

	// TTS defaults: the voice curds asks for when -voice is omitted, and the
	// per-model text caps enforced before any network call.
	DefaultTTSVoice           = "English_Wiselady" // minimax/speech-2.8-hd
	DefaultTTSGeminiVoice     = "Kore"             // google/gemini-3.1-flash-tts
	DefaultTTSElevenLabsVoice = "Rachel"
	DefaultTTSOpenAIVoice     = "sage"
	MaxTTSTextChars           = 10000 // minimax/speech-2.8-hd
	MaxGeminiTTSBytes         = 4000  // google/gemini-3.1-flash-tts (text and prompt each)
	MaxOpenAITTSChars         = 4096  // openai POST /v1/audio/speech

	// FluxImageModel is FLUX.2 [pro] on Replicate: fast, cheap image generation
	// with reference-image control. Uses megapixel resolutions, not sizes.
	FluxImageModel = "black-forest-labs/flux-2-pro"

	// NanoBananaImageModel is Google Nano Banana 2 on Replicate: strong
	// character consistency and multi-image compositing, 1K/2K/4K output.
	NanoBananaImageModel = "google/nano-banana-2"

	// TopazUpscaleModel is Topaz Labs' image upscaler on Replicate: 2x/4x/6x
	// factors plus face enhancement. Selectable via -model upscale-pro.
	TopazUpscaleModel = "topazlabs/image-upscale"

	// DefaultXaiVideoModel is xAI's native Grok Imagine Video model id, used
	// when routing video through the x.ai API directly instead of Replicate.
	DefaultXaiVideoModel = "grok-imagine-video"

	// DefaultSegmentationModel is the Replicate-hosted background-removal /
	// segmentation model used when -model remove-bg is requested. BRIA RMBG
	// 2.0 is an official Replicate model (no version pin), takes a single
	// `image` URL and returns a transparent PNG with a 256-level alpha matte.
	DefaultSegmentationModel = "bria/remove-background"

	// DefaultUpscaleModel is the Replicate-hosted super-resolution model used
	// when -model upscale is requested. prunaai/p-image-upscale scales each
	// side by a factor (or to a target megapixel count) and returns one image
	// in png/jpg/webp. It generates no new pixels from a prompt, so it shares
	// the segmentation request/validation path.
	DefaultUpscaleModel = "prunaai/p-image-upscale"

	// RealESRGANUpscaleModel is nightmareai/real-esrgan (2021-era): a single
	// `image` URL plus a `scale` factor and optional `face_enhance`, returning
	// one upscaled PNG. The one upscaler with face enhancement. Selectable via
	// -model upscale-esrgan.
	RealESRGANUpscaleModel = "nightmareai/real-esrgan"

	// DefaultUpscaleScale is the default super-resolution factor sent when the
	// caller leaves Request.Scale at zero. Both upscalers accept 4.
	DefaultUpscaleScale = 4

	// MaxPrunaUpscaleFactor is prunaai/p-image-upscale's per-side factor cap.
	MaxPrunaUpscaleFactor = 8

	// MaxESRGANUpscaleFactor is nightmareai/real-esrgan's scale cap.
	MaxESRGANUpscaleFactor = 10

	MaxInputImages = 16

	defaultPollInterval = 2 * time.Second
)

// AspectRatioSizes maps friendly aspect ratios to OpenAI-compatible pixel
// sizes. All edges are multiples of 16 and total pixels stay within
// gpt-image-2's allowed range. 16:9/9:16 land slightly above 1080p because
// 1080 is not a multiple of 16.
var AspectRatioSizes = map[string]string{
	"1:1":     "1024x1024",
	"3:2":     "1536x1024",
	"2:3":     "1024x1536",
	"4:3":     "1536x1152",
	"3:4":     "1152x1536",
	"16:9":    "2048x1152",
	"9:16":    "1152x2048",
	"21:9":    "2688x1152",
	"9:21":    "1152x2688",
	"2:1":     "2048x1024",
	"1:2":     "1024x2048",
	"16:9-4k": "3840x2160",
	"9:16-4k": "2160x3840",
}

// ReplicateAllowedAspectRatios are the aspect_ratio values supported by
// Replicate's openai/gpt-image-2 wrapper.
var ReplicateAllowedAspectRatios = map[string]bool{
	"1:1": true, "3:2": true, "2:3": true,
}

// Request describes a single generation call.
type Request struct {
	Provider          string
	Token             string
	Model             string // empty = provider default
	Prompt            string
	AspectRatio       string // e.g. "16:9"; ignored if Size is set
	Size              string // e.g. "2048x1152"; OpenAI only
	Quality           string // low, medium, high, xhigh, max, auto
	NumImages         int
	OutputFormat      string // webp, png, jpeg, mp4
	OutputCompression int    // 0-100; OpenAI webp/jpeg only
	Background        string // auto, opaque
	Moderation        string // auto, low
	User              string // OpenAI only
	ReplicateBYOKey   string // optional OpenAI key passed through Replicate
	InputImages       []string
	Mask              string // OpenAI edits only
	LastFrameImage    string // Replicate video only
	ReferenceImages   []string
	ReferenceVideos   []string
	ReferenceAudios   []string
	ImageResolution   string  // flux: 0.5mp/1mp/2mp/4mp; nano-banana: 1k/2k/4k
	VideoDuration     int     // seconds; model-specific, 0 = default
	VideoResolution   string  // 480p, 720p, 1080p; empty = default
	GenerateAudio     *bool   // nil = provider default
	Seed              int     // 0 = provider random seed
	Scale             float64 // upscale factor for super-resolution models; 0 = model default
	// Duration is the requested length in seconds for audio models. music maps
	// it onto music_length_ms (5-300), sfx onto its integer duration (1-190,
	// default 10), and music-vocal ignores it upstream — curds trims the
	// render locally instead. 0 = model default.
	Duration float64
	// Instrumental requests an instrumental track (no vocals) from the music
	// models. nil = model default: true for -model music (ElevenLabs Music is
	// instrumental-first) and false for -model music-vocal. Non-empty Lyrics
	// force it false, because lyrics imply vocals. Ignored by -model sfx.
	Instrumental *bool
	// Lyrics is the song text for minimax/music-2.6 (-lyrics TEXT or
	// -lyrics @file.txt); [Verse]/[Chorus] tags and newlines pass through.
	// Only -model music-vocal accepts it.
	Lyrics string
	// Voice is the voice for a TTS model. minimax/speech-2.8-hd accepts any
	// system voice id or a cloned id (free-form); Gemini 3.1 Flash TTS,
	// elevenlabs/v3 and the OpenAI speech models take a name from their own
	// enum. Empty = model default.
	Voice string
	// Emotion is minimax/speech-2.8-hd's delivery emotion (auto, happy, sad,
	// angry, fearful, disgusted, surprised, calm, fluent, neutral). Empty =
	// auto. Rejected by the other TTS models.
	Emotion string
	// Speed is the TTS speaking rate. 0 = the model's own default; each model
	// has its own range (0.5-2, 0.7-1.2, or 0.25-4). Gemini 3.1 Flash TTS has
	// no speed knob, so it rejects the flag; steer pace via Instructions.
	Speed float64
	// Pitch shifts minimax/speech-2.8-hd's voice in semitones, -12..12.
	// 0 = unshifted. Rejected by the other TTS models.
	Pitch int
	// Instructions steers delivery: Gemini 3.1 Flash TTS's style prompt
	// (tone, pace, accent, character) and OpenAI gpt-4o-mini-tts's
	// instructions ("crisp British RP, dry"). Rejected by every other TTS
	// model.
	Instructions string
	// Stability is elevenlabs/v3's voice stability, 0-1. nil = the model's own
	// default (0.5). Rejected by the other TTS models.
	Stability *float64
	// Style is elevenlabs/v3's style exaggeration, 0-1. nil = the model's own
	// default (0). Rejected by the other TTS models.
	Style *float64
	// Audio is the audio track for talking-head models (kling-avatar,
	// lipsync): a file path, http(s) URL, or data URL.
	Audio string
	// InputVideo is the source video for lipsync models (file path, URL, or
	// data URL).
	InputVideo string
	// SyncMode selects how lipsync-2-pro loops the source video: loop, bounce,
	// cut_off, silence, or remap. Empty = the provider default (loop).
	SyncMode string
	// SyncTemperature is lipsync-2-pro's temperature, 0-1. A negative value
	// means "provider default" (0.5) and omits the field.
	SyncTemperature float64
	// ActiveSpeaker tells lipsync-2-pro to animate the active speaker in the
	// source video.
	ActiveSpeaker bool
	// NoFallback disables the automatic Seedance→Kling 3.0 retry when
	// Seedance rejects a realistic human face as sensitive.
	NoFallback bool

	FaceEnhance bool // run GFPGAN face enhancement (upscale models only)

	PollInterval time.Duration // Replicate poll cadence; 0 = default
	Logger       io.Writer     // logfmt event sink (info/error always written when set)
	Verbose      bool          // when true, debug-level events are emitted as well
}

// Image is one rendered image.
type Image struct {
	Bytes         []byte
	Format        string
	RevisedPrompt string // populated for OpenAI when available
}

// Video is one rendered video.
type Video struct {
	Bytes  []byte
	Format string
	URL    string
}

// Audio is one rendered audio asset. Format is the container the provider
// actually returned (mp3 or wav), which for a model that ignores the container
// request can differ from Request.OutputFormat.
type Audio struct {
	Bytes  []byte
	Format string
	URL    string
}

// Result groups all assets produced by a Request.
type Result struct {
	Images []Image
	Videos []Video
	Audios []Audio
}

// Provider is the contract every backend implements.
type Provider interface {
	Generate(ctx context.Context, req *Request) (*Result, error)
	Name() string
}

// Client routes a Request to the appropriate Provider. Provider fields are
// pluggable so callers can substitute fakes in tests.
type Client struct {
	HTTPClient *http.Client
	Replicate  Provider
	OpenAI     Provider
	Xai        Provider
}

// New constructs a Client with default providers wired to the public APIs.
func New() *Client {
	hc := &http.Client{Timeout: 0}
	return &Client{
		HTTPClient: hc,
		Replicate:  &ReplicateProvider{HTTPClient: hc, APIBase: "https://api.replicate.com/v1"},
		OpenAI:     &OpenAIProvider{HTTPClient: hc, APIBase: "https://api.openai.com/v1"},
		Xai:        &XaiProvider{HTTPClient: hc, APIBase: "https://api.x.ai/v1"},
	}
}

// Generate validates the request and dispatches to the chosen provider.
func (c *Client) Generate(ctx context.Context, req *Request) (*Result, error) {
	req.applyDefaults()
	if err := req.Validate(); err != nil {
		return nil, err
	}
	p, err := c.providerFor(req.Provider)
	if err != nil {
		return nil, err
	}
	return p.Generate(ctx, req)
}

func (c *Client) providerFor(name string) (Provider, error) {
	switch name {
	case ProviderReplicate:
		if c.Replicate == nil {
			return nil, errors.New("replicate provider not configured")
		}
		return c.Replicate, nil
	case ProviderOpenAI:
		if c.OpenAI == nil {
			return nil, errors.New("openai provider not configured")
		}
		return c.OpenAI, nil
	case ProviderXai:
		if c.Xai == nil {
			return nil, errors.New("xai provider not configured")
		}
		return c.Xai, nil
	}
	return nil, fmt.Errorf("unsupported provider %q", name)
}

func (r *Request) applyDefaults() {
	if r.Model == "" {
		r.Model = DefaultModel(r.Provider)
	}
	if r.NumImages == 0 {
		r.NumImages = 1
	}
	if r.AspectRatio == "" && r.Size == "" {
		switch {
		case IsAudioModel(r.Model):
			// Audio models take no aspect ratio: the model decides the mix.
		case IsGrokImagineVideoModel(r.Model), IsXaiVideoModel(r.Model):
			r.AspectRatio = "auto"
		case IsMinimaxVideoModel(r.Model), IsKlingVideoModel(r.Model):
			// Neither accepts "auto", and 1:1 is a poor video default.
			r.AspectRatio = "16:9"
		default:
			r.AspectRatio = "1:1"
		}
	}
	if r.Quality == "" {
		r.Quality = "auto"
	}
	if r.OutputFormat == "" {
		switch {
		case IsVideoModel(r.Model):
			r.OutputFormat = "mp4"
		case IsAudioModel(r.Model):
			r.OutputFormat = "mp3"
		case IsSegmentationModel(r.Model), IsUpscaleModel(r.Model):
			r.OutputFormat = "png"
		case IsNanoBananaImageModel(r.Model):
			// Nano Banana 2 emits jpg or png only — no webp.
			r.OutputFormat = "png"
		default:
			r.OutputFormat = "webp"
		}
	}
	if IsVideoModel(r.Model) && r.VideoResolution == "" {
		switch {
		case IsLipsyncModel(r.Model):
			// lipsync-2-pro has no resolution knob: the source video's own
			// resolution is what comes back.
		case IsMinimaxVideoModel(r.Model):
			// H3's vocabulary is 768P / 2K; default to the cheaper tier.
			r.VideoResolution = "768p"
		case IsKlingVideoModel(r.Model), IsKlingAvatarModel(r.Model):
			// Matches Kling's own default mode ("pro"; 1080p for the avatar).
			r.VideoResolution = "1080p"
		default:
			r.VideoResolution = "720p"
		}
	}
	if IsAudioModel(r.Model) {
		// Stable Audio's own default length (10s) is what curds asks for when
		// -duration is omitted; the music models default upstream.
		if IsSFXModel(r.Model) && r.Duration == 0 {
			r.Duration = DefaultSFXDuration
		}
		// Lyrics imply vocals, so they override -instrumental. Otherwise fill
		// in the model's own default: instrumental for ElevenLabs Music,
		// vocal for MiniMax Music 2.6.
		switch {
		case strings.TrimSpace(r.Lyrics) != "":
			instrumental := false
			r.Instrumental = &instrumental
		case r.Instrumental == nil && IsMusicModel(r.Model):
			instrumental := true
			r.Instrumental = &instrumental
		case r.Instrumental == nil && IsMusicVocalModel(r.Model):
			instrumental := false
			r.Instrumental = &instrumental
		}
		if IsTTSModel(r.Model) && r.Voice == "" {
			r.Voice = DefaultVoiceFor(r.Model)
		}
	}
	if r.Background == "" {
		r.Background = "auto"
	}
	if r.Moderation == "" {
		r.Moderation = "auto"
	}
	if r.PollInterval == 0 {
		r.PollInterval = defaultPollInterval
	}
}

// Validate checks the request for obvious errors before dispatch.
func (r *Request) Validate() error {
	if r.Provider == "" {
		return errors.New("provider is required")
	}
	switch r.Provider {
	case ProviderReplicate, ProviderOpenAI, ProviderXai:
	default:
		return fmt.Errorf("unsupported provider %q (supported: openai, replicate, xai)", r.Provider)
	}
	if err := CheckProviderModel(r.Provider, r.Model); err != nil {
		return err
	}
	if r.Token == "" {
		return fmt.Errorf("missing %s token", r.Provider)
	}
	// MiniMax Music 2.6 can also be driven by -lyrics alone (lyrics imply
	// vocals), so it is the one prompt-required model with an alternative.
	if !IsPromptlessModel(r.Model) && strings.TrimSpace(r.Prompt) == "" && !r.lyricsOnlyMusic() {
		return errors.New("prompt is required")
	}
	if r.NumImages < 1 || r.NumImages > 10 {
		return fmt.Errorf("num_images must be 1-10, got %d", r.NumImages)
	}
	if r.OutputCompression < 0 || r.OutputCompression > 100 {
		return fmt.Errorf("output_compression must be 0-100, got %d", r.OutputCompression)
	}
	if max := MaxInputImagesFor(r.Model); len(r.InputImages) > max {
		return fmt.Errorf("at most %d input images supported, got %d", max, len(r.InputImages))
	}
	switch {
	case IsVideoModel(r.Model):
		if err := r.validateVideo(); err != nil {
			return err
		}
	case IsTTSModel(r.Model):
		if err := r.validateTTS(); err != nil {
			return err
		}
	case IsAudioModel(r.Model):
		if err := r.validateAudio(); err != nil {
			return err
		}
	case IsSegmentationModel(r.Model):
		if err := r.validateSegmentation(); err != nil {
			return err
		}
	case IsUpscaleModel(r.Model):
		if err := r.validateUpscale(); err != nil {
			return err
		}
	case IsFluxImageModel(r.Model):
		if err := r.validateFluxImage(); err != nil {
			return err
		}
	case IsNanoBananaImageModel(r.Model):
		if err := r.validateNanoBananaImage(); err != nil {
			return err
		}
	default:
		switch r.OutputFormat {
		case "webp", "png", "jpeg":
		default:
			return fmt.Errorf("output_format must be webp, png, or jpeg, got %q", r.OutputFormat)
		}
	}
	// The gpt-image-2 wrapper on Replicate is the only model with the narrow
	// 1:1/3:2/2:3 ratio list; models with their own ratio enums validate above.
	if r.Provider == ProviderReplicate && !IsVideoModel(r.Model) && !IsAudioModel(r.Model) &&
		!IsSegmentationModel(r.Model) &&
		!IsUpscaleModel(r.Model) && !IsFluxImageModel(r.Model) && !IsNanoBananaImageModel(r.Model) &&
		r.AspectRatio != "" && !ReplicateAllowedAspectRatios[r.AspectRatio] {
		return fmt.Errorf("replicate only supports 1:1, 3:2, 2:3 aspect ratios; got %q", r.AspectRatio)
	}
	if r.Provider == ProviderOpenAI && r.Size == "" && r.AspectRatio != "" && r.AspectRatio != "auto" {
		if _, ok := AspectRatioSizes[r.AspectRatio]; !ok {
			return fmt.Errorf("unknown aspect ratio %q; pass -size WxH or one of: %v", r.AspectRatio, SortedAspectRatios())
		}
	}
	if r.Provider == ProviderReplicate {
		if idx := strings.Index(r.Model, ":"); idx >= 0 {
			if strings.TrimSpace(r.Model[idx+1:]) == "" {
				return fmt.Errorf("model %q has empty version after ':'", r.Model)
			}
		}
	}
	return nil
}

func (r *Request) validateVideo() error {
	if r.NumImages != 1 {
		return fmt.Errorf("video generation supports exactly one output, got num_images=%d", r.NumImages)
	}
	if r.OutputFormat != "mp4" {
		return fmt.Errorf("video output_format must be mp4, got %q", r.OutputFormat)
	}
	if r.Size != "" {
		return errors.New("video generation uses -aspect-ratio and -video-resolution; -size is image-only")
	}
	if r.Mask != "" {
		return errors.New("video generation does not support -mask")
	}
	switch {
	case IsXaiVideoModel(r.Model):
		if r.Provider != ProviderXai {
			return fmt.Errorf("model %q is only supported with provider xai", r.Model)
		}
		return r.validateXaiVideo()
	case IsGrokImagineVideoModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateGrokImagineVideo()
	case IsSeedanceModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateSeedanceVideo()
	case IsMinimaxVideoModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateMinimaxVideo()
	case IsKlingVideoModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateKlingVideo()
	case IsKlingAvatarModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateKlingAvatarVideo()
	case IsLipsyncModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		return r.validateLipsyncVideo()
	default:
		return fmt.Errorf("unsupported video model %q", r.Model)
	}
}

// AudioAllowedFormats are the containers the audio models can emit, and the
// values -output-format accepts for them.
var AudioAllowedFormats = map[string]bool{"mp3": true, "wav": true}

// lyricsOnlyMusic reports whether the request is a minimax/music-2.6 song
// driven by -lyrics alone. Lyrics imply vocals, so the prompt requirement is
// satisfied upstream by the lyrics themselves. Only that one model qualifies.
func (r *Request) lyricsOnlyMusic() bool {
	return IsMusicVocalModel(r.Model) && strings.TrimSpace(r.Lyrics) != ""
}

// validateAudio checks a request for the audio models (ElevenLabs Music,
// MiniMax Music 2.6, Stable Audio 2.5). They are prompt-in/audio-out on
// Replicate: no input media, no pixel size, no aspect ratio, one file per
// request. Each accepts its own duration range, and -lyrics belongs to
// music-vocal alone.
func (r *Request) validateAudio() error {
	if r.Provider != ProviderReplicate {
		return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
	}
	if r.NumImages != 1 {
		return fmt.Errorf("audio generation produces exactly one file, got num_images=%d", r.NumImages)
	}
	if !AudioAllowedFormats[r.OutputFormat] {
		return fmt.Errorf("audio output_format must be mp3 or wav, got %q", r.OutputFormat)
	}
	if r.Size != "" {
		return errors.New("audio generation has no pixel size; -size is image-only")
	}
	if r.Mask != "" {
		return errors.New("audio generation does not accept -mask")
	}
	if len(r.InputImages) > 0 || r.LastFrameImage != "" || r.InputVideo != "" ||
		len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("audio generation takes no input media (only -prompt, plus -lyrics for -model music-vocal)")
	}
	switch {
	case IsMusicModel(r.Model):
		if strings.TrimSpace(r.Lyrics) != "" {
			return errors.New("-lyrics is only supported by -model music-vocal (ElevenLabs Music renders instrumentals)")
		}
		if r.Duration != 0 && (r.Duration < 5 || r.Duration > 300) {
			return fmt.Errorf("duration must be 5-300 seconds for ElevenLabs Music, got %g", r.Duration)
		}
	case IsMusicVocalModel(r.Model):
		if strings.TrimSpace(r.Prompt) == "" && strings.TrimSpace(r.Lyrics) == "" {
			return errors.New("MiniMax Music 2.6 needs -prompt, or -lyrics with -instrumental=false")
		}
		if r.Duration != 0 && r.Duration < 1 {
			return fmt.Errorf("duration must be at least 1 second for MiniMax Music 2.6, got %g", r.Duration)
		}
	case IsSFXModel(r.Model):
		if strings.TrimSpace(r.Lyrics) != "" {
			return errors.New("-lyrics is only supported by -model music-vocal")
		}
		if r.Duration != 0 && (r.Duration < 1 || r.Duration > 190) {
			return fmt.Errorf("duration must be 1-190 seconds for Stable Audio 2.5, got %g", r.Duration)
		}
	default:
		return fmt.Errorf("unsupported audio model %q", r.Model)
	}
	if r.Seed != 0 && !IsSFXModel(r.Model) {
		return errors.New("this audio model has no seed input; -seed is supported by -model sfx only")
	}
	return nil
}

// GeminiTTSVoices are the voice names google/gemini-3.1-flash-tts accepts.
// curds validates -voice against this list so a typo fails locally instead of
// upstream. The upstream default is Kore.
var GeminiTTSVoices = map[string]bool{
	"Achernar": true, "Achird": true, "Algenib": true, "Algieba": true,
	"Alnilam": true, "Aoede": true, "Autonoe": true, "Callirrhoe": true,
	"Charon": true, "Despina": true, "Enceladus": true, "Erinome": true,
	"Fenrir": true, "Gacrux": true, "Iapetus": true, "Kore": true,
	"Laomedeia": true, "Leda": true, "Orus": true, "Pulcherrima": true,
	"Puck": true, "Rasalgethi": true, "Sadachbia": true, "Sadaltager": true,
	"Schedar": true, "Sulafat": true, "Umbriel": true, "Vindemiatrix": true,
	"Zephyr": true, "Zubenelgenubi": true,
}

// ElevenLabsTTSVoices are the voice names elevenlabs/v3 accepts. curds
// validates -voice against this list so a typo fails locally instead of
// upstream.
var ElevenLabsTTSVoices = map[string]bool{
	"Rachel": true, "Drew": true, "Clyde": true, "Paul": true, "Aria": true,
	"Domi": true, "Dave": true, "Roger": true, "Fin": true, "Sarah": true,
	"James": true, "Jane": true, "Juniper": true, "Arabella": true,
	"Hope": true, "Bradford": true, "Reginald": true, "Gaming": true,
	"Austin": true, "Kuon": true, "Blondie": true, "Priyanka": true,
	"Alexandra": true, "Monika": true, "Mark": true, "Grimblewood": true,
}

// OpenAITTSVoices are the voice names POST /v1/audio/speech accepts.
var OpenAITTSVoices = map[string]bool{
	"alloy": true, "ash": true, "ballad": true, "coral": true, "echo": true,
	"fable": true, "onyx": true, "nova": true, "sage": true, "shimmer": true,
	"verse": true, "marin": true, "cedar": true,
}

// MinimaxTTSEmotions are the delivery emotions minimax/speech-2.8-hd accepts.
var MinimaxTTSEmotions = map[string]bool{
	"auto": true, "happy": true, "sad": true, "angry": true, "fearful": true,
	"disgusted": true, "surprised": true, "calm": true, "fluent": true,
	"neutral": true,
}

// DefaultVoiceFor returns the voice curds asks a TTS model for when -voice is
// omitted: Gemini's Kore, MiniMax's warm English system voice, ElevenLabs'
// Rachel, or OpenAI's sage.
func DefaultVoiceFor(model string) string {
	switch {
	case IsTTSGeminiModel(model):
		return DefaultTTSGeminiVoice
	case IsTTSElevenLabsModel(model):
		return DefaultTTSElevenLabsVoice
	case IsOpenAITTSModel(model):
		return DefaultTTSOpenAIVoice
	default:
		return DefaultTTSVoice
	}
}

// SortedNames renders an enum map as a sorted, comma-separated list for error
// messages, so a rejected -voice tells the user what it could have been.
func SortedNames(voices map[string]bool) string {
	names := make([]string, 0, len(voices))
	for name := range voices {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// validateTTS checks a text-to-speech request. The four models share the
// audio shape (one file, mp3/wav, no input media) but differ in provider,
// voices, speed range, and which delivery knobs they take, so each is checked
// against its own contract. Flags belonging to another TTS model are rejected
// rather than silently ignored.
func (r *Request) validateTTS() error {
	if r.NumImages != 1 {
		return fmt.Errorf("text-to-speech produces exactly one file, got num_images=%d", r.NumImages)
	}
	if !AudioAllowedFormats[r.OutputFormat] {
		return fmt.Errorf("audio output_format must be mp3 or wav, got %q", r.OutputFormat)
	}
	if r.Size != "" {
		return errors.New("text-to-speech has no pixel size; -size is image-only")
	}
	if r.Mask != "" {
		return errors.New("text-to-speech does not accept -mask")
	}
	if len(r.InputImages) > 0 || r.LastFrameImage != "" || r.InputVideo != "" ||
		len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("text-to-speech takes no input media; it reads -prompt (or stdin)")
	}
	switch {
	case IsTTSGeminiModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		if r.Voice != "" && !GeminiTTSVoices[r.Voice] {
			return fmt.Errorf("voice must be one of %s, got %q", SortedNames(GeminiTTSVoices), r.Voice)
		}
		if len(r.Prompt) > MaxGeminiTTSBytes {
			return fmt.Errorf("text is %d bytes; Gemini 3.1 Flash TTS accepts at most %d", len(r.Prompt), MaxGeminiTTSBytes)
		}
		if len(r.Instructions) > MaxGeminiTTSBytes {
			return fmt.Errorf("-instructions is %d bytes; Gemini 3.1 Flash TTS's style prompt accepts at most %d", len(r.Instructions), MaxGeminiTTSBytes)
		}
		if r.Speed != 0 {
			return errors.New("-speed is not supported by -model tts (Gemini 3.1 Flash TTS); steer pace with -instructions")
		}
		if r.Emotion != "" {
			return errors.New("-emotion is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Pitch != 0 {
			return errors.New("-pitch is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Stability != nil || r.Style != nil {
			return errors.New("-stability and -style are only supported by -model tts-elevenlabs")
		}
	case IsTTSSpeechModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		if utf8.RuneCountInString(r.Prompt) > MaxTTSTextChars {
			return fmt.Errorf("text is %d characters; MiniMax Speech 2.8 HD accepts at most %d", utf8.RuneCountInString(r.Prompt), MaxTTSTextChars)
		}
		if r.Emotion != "" && !MinimaxTTSEmotions[r.Emotion] {
			return fmt.Errorf("emotion must be one of %s, got %q", SortedNames(MinimaxTTSEmotions), r.Emotion)
		}
		if r.Speed != 0 && (r.Speed < 0.5 || r.Speed > 2) {
			return fmt.Errorf("speed must be 0.5-2 for MiniMax Speech 2.8 HD, got %g", r.Speed)
		}
		if r.Pitch < -12 || r.Pitch > 12 {
			return fmt.Errorf("pitch must be -12..12 for MiniMax Speech 2.8 HD, got %d", r.Pitch)
		}
		if r.Stability != nil || r.Style != nil {
			return errors.New("-stability and -style are only supported by -model tts-elevenlabs")
		}
		if r.Instructions != "" {
			return errors.New("-instructions is only supported by -model tts (Gemini 3.1 Flash TTS) and -model tts-openai (gpt-4o-mini-tts)")
		}
	case IsTTSElevenLabsModel(r.Model):
		if r.Provider != ProviderReplicate {
			return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
		}
		if r.Voice != "" && !ElevenLabsTTSVoices[r.Voice] {
			return fmt.Errorf("voice must be one of %s, got %q", SortedNames(ElevenLabsTTSVoices), r.Voice)
		}
		if r.Speed != 0 && (r.Speed < 0.7 || r.Speed > 1.2) {
			return fmt.Errorf("speed must be 0.7-1.2 for ElevenLabs v3, got %g", r.Speed)
		}
		if r.Stability != nil && (*r.Stability < 0 || *r.Stability > 1) {
			return fmt.Errorf("stability must be 0-1 for ElevenLabs v3, got %g", *r.Stability)
		}
		if r.Style != nil && (*r.Style < 0 || *r.Style > 1) {
			return fmt.Errorf("style must be 0-1 for ElevenLabs v3, got %g", *r.Style)
		}
		if r.Emotion != "" {
			return errors.New("-emotion is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Pitch != 0 {
			return errors.New("-pitch is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Instructions != "" {
			return errors.New("-instructions is only supported by -model tts (Gemini 3.1 Flash TTS) and -model tts-openai (gpt-4o-mini-tts)")
		}
	case IsOpenAITTSModel(r.Model):
		if r.Provider != ProviderOpenAI {
			return fmt.Errorf("model %q is only supported with provider openai", r.Model)
		}
		if r.Voice != "" && !OpenAITTSVoices[r.Voice] {
			return fmt.Errorf("voice must be one of %s, got %q", SortedNames(OpenAITTSVoices), r.Voice)
		}
		if r.Speed != 0 && (r.Speed < 0.25 || r.Speed > 4) {
			return fmt.Errorf("speed must be 0.25-4 for OpenAI speech, got %g", r.Speed)
		}
		if r.Instructions != "" && !IsOpenAITTSMiniModel(r.Model) {
			return fmt.Errorf("-instructions is only supported by gpt-4o-mini-tts, not %s", r.Model)
		}
		if utf8.RuneCountInString(r.Prompt) > MaxOpenAITTSChars {
			return fmt.Errorf("text is %d characters; OpenAI speech accepts at most %d", utf8.RuneCountInString(r.Prompt), MaxOpenAITTSChars)
		}
		if r.Emotion != "" {
			return errors.New("-emotion is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Pitch != 0 {
			return errors.New("-pitch is only supported by -model tts-minimax (MiniMax Speech 2.8 HD)")
		}
		if r.Stability != nil || r.Style != nil {
			return errors.New("-stability and -style are only supported by -model tts-elevenlabs")
		}
	default:
		return fmt.Errorf("unsupported text-to-speech model %q", r.Model)
	}
	if r.Seed != 0 {
		return errors.New("text-to-speech models have no seed input")
	}
	if r.Duration != 0 {
		return errors.New("text-to-speech has no -duration; the text sets the length")
	}
	if r.Instrumental != nil {
		return errors.New("-instrumental is only supported by the music models")
	}
	if strings.TrimSpace(r.Lyrics) != "" {
		return errors.New("-lyrics is only supported by -model music-vocal")
	}
	return nil
}

// validateXaiVideo checks a request for xAI's native Grok Imagine Video API.
// Unlike the Replicate wrapper this supports text-to-video (image optional),
// and reference images. The source image goes via -input-image (0 or 1)
// and additional references via -reference-image.
func (r *Request) validateXaiVideo() error {
	if len(r.InputImages) > 1 {
		return fmt.Errorf("xai video accepts at most one -input-image source (use -reference-image for additional references), got %d", len(r.InputImages))
	}
	if r.VideoDuration != 0 && (r.VideoDuration < 1 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be 1-15 seconds for grok-imagine-video, got %d", r.VideoDuration)
	}
	switch r.VideoResolution {
	case "480p", "720p":
	default:
		return fmt.Errorf("video_resolution must be 480p or 720p for native grok-imagine-video, got %q; choose -video-resolution 720p explicitly, or use -provider replicate -model seedance-2 for 1080p; no automatic resolution or model change is made", r.VideoResolution)
	}
	switch r.AspectRatio {
	case "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3":
	default:
		return fmt.Errorf("grok-imagine-video aspect_ratio must be auto, 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, or 2:3; got %q", r.AspectRatio)
	}
	if r.LastFrameImage != "" {
		return errors.New("xai video does not support -last-frame-image")
	}
	if len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("xai video does not support -reference-video or -reference-audio")
	}
	if r.Seed != 0 {
		return errors.New("xai video does not support -seed")
	}
	if r.GenerateAudio != nil && !*r.GenerateAudio {
		return errors.New("xai video generates audio automatically and does not support -no-audio")
	}
	return nil
}

func (r *Request) validateGrokImagineVideo() error {
	if len(r.InputImages) != 1 {
		return fmt.Errorf("Grok Imagine Video 1.5 requires exactly one -input-image, got %d", len(r.InputImages))
	}
	if r.VideoDuration != 0 && (r.VideoDuration < 1 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be 1-15 seconds for Grok Imagine Video 1.5, got %d", r.VideoDuration)
	}
	switch r.VideoResolution {
	case "480p", "720p":
	default:
		return fmt.Errorf("video_resolution must be 480p or 720p for Grok Imagine Video 1.5, got %q", r.VideoResolution)
	}
	switch r.AspectRatio {
	case "auto", "16:9", "4:3", "1:1", "9:16", "3:4", "3:2", "2:3":
	default:
		return fmt.Errorf("grok-imagine-video-1.5 aspect_ratio must be auto, 16:9, 4:3, 1:1, 9:16, 3:4, 3:2, or 2:3; got %q", r.AspectRatio)
	}
	if r.LastFrameImage != "" {
		return errors.New("Grok Imagine Video 1.5 does not support -last-frame-image")
	}
	if len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("Grok Imagine Video 1.5 does not support -reference-image, -reference-video, or -reference-audio")
	}
	if r.Seed != 0 {
		return errors.New("Grok Imagine Video 1.5 does not support -seed")
	}
	if r.GenerateAudio != nil && !*r.GenerateAudio {
		return errors.New("Grok Imagine Video 1.5 generates audio automatically and does not support -no-audio")
	}
	return nil
}

// validateKlingVideo checks a request for Kling Video 3.0 (Replicate). Kling
// does text-to-video or image-to-video with a start frame and optional end
// frame, 3-15s, in standard (720p) / pro (1080p) / 4k modes, with native audio.
func (r *Request) validateKlingVideo() error {
	if r.VideoDuration != 0 && (r.VideoDuration < 3 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be 3-15 seconds for Kling 3.0, got %d", r.VideoDuration)
	}
	if KlingMode(r.VideoResolution) == "" {
		return fmt.Errorf("video_resolution must be 720p, 1080p, or 4k for Kling 3.0, got %q", r.VideoResolution)
	}
	switch r.AspectRatio {
	case "16:9", "9:16", "1:1":
	default:
		return fmt.Errorf("Kling 3.0 aspect_ratio must be 16:9, 9:16, or 1:1; got %q", r.AspectRatio)
	}
	if len(r.InputImages) > 1 {
		return fmt.Errorf("Kling 3.0 accepts at most one -input-image start frame, got %d", len(r.InputImages))
	}
	if r.LastFrameImage != "" && len(r.InputImages) != 1 {
		return errors.New("-last-frame-image requires exactly one -input-image start frame")
	}
	if len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("Kling 3.0 does not support -reference-image, -reference-video, or -reference-audio; use -model seedance-2 for reference-guided video")
	}
	if r.Seed != 0 {
		return errors.New("Kling 3.0 does not support -seed")
	}
	return nil
}

// KlingMode maps a curds -video-resolution value onto Kling's mode enum,
// returning "" when the value is not one Kling accepts.
func KlingMode(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "720p":
		return "standard"
	case "1080p":
		return "pro"
	case "4k":
		return "4k"
	}
	return ""
}

// validateKlingAvatarVideo checks a request for Kling Avatar 2.0 (Replicate).
// It animates exactly one portrait with exactly one audio clip; the prompt is
// optional (actions/emotion/camera) and `-video-resolution` maps onto the
// model's mode enum: 720p = std, 1080p = pro (curds' default).
func (r *Request) validateKlingAvatarVideo() error {
	if len(r.InputImages) != 1 {
		return fmt.Errorf("Kling Avatar 2.0 requires exactly one -input-image portrait, got %d", len(r.InputImages))
	}
	if strings.TrimSpace(r.Audio) == "" {
		return errors.New("Kling Avatar 2.0 requires -audio PATH (mp3, wav, m4a, or aac)")
	}
	if KlingAvatarMode(r.VideoResolution) == "" {
		return fmt.Errorf("video_resolution must be 720p or 1080p for Kling Avatar 2.0, got %q", r.VideoResolution)
	}
	if r.InputVideo != "" {
		return errors.New("Kling Avatar 2.0 does not take -input-video; use -model lipsync to drive an existing video")
	}
	if r.LastFrameImage != "" || len(r.ReferenceImages) > 0 ||
		len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("Kling Avatar 2.0 animates one portrait and one audio clip; it does not support -last-frame-image or references")
	}
	if r.Seed != 0 {
		return errors.New("Kling Avatar 2.0 does not support -seed")
	}
	return nil
}

// KlingAvatarMode maps a curds -video-resolution value onto Kling Avatar 2.0's
// mode enum, returning "" when the value is not one it accepts.
func KlingAvatarMode(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "720p":
		return "std"
	case "1080p":
		return "pro"
	}
	return ""
}

// LipsyncAllowedSyncModes is sync/lipsync-2-pro's sync_mode enum: how the model
// handles audio longer than the source video.
var LipsyncAllowedSyncModes = map[string]bool{
	"loop": true, "bounce": true, "cut_off": true, "silence": true, "remap": true,
}

// validateLipsyncVideo checks a request for sync/lipsync-2-pro (Replicate). It
// re-animates the mouth in an existing video to match an audio clip, so it
// needs -input-video and -audio and takes no prompt.
func (r *Request) validateLipsyncVideo() error {
	if strings.TrimSpace(r.InputVideo) == "" {
		return errors.New("lipsync requires -input-video PATH (mp4)")
	}
	if strings.TrimSpace(r.Audio) == "" {
		return errors.New("lipsync requires -audio PATH (wav)")
	}
	if strings.TrimSpace(r.Prompt) != "" {
		return errors.New("lipsync does not take a prompt; drop -prompt (the source video drives the animation)")
	}
	if mode := strings.ToLower(strings.TrimSpace(r.SyncMode)); mode != "" && !LipsyncAllowedSyncModes[mode] {
		return fmt.Errorf("sync_mode must be loop, bounce, cut_off, silence, or remap; got %q", r.SyncMode)
	}
	if r.SyncTemperature > 1 {
		return fmt.Errorf("sync_temperature must be 0-1, got %v", r.SyncTemperature)
	}
	if len(r.InputImages) > 0 || r.LastFrameImage != "" || len(r.ReferenceImages) > 0 ||
		len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("lipsync drives -input-video through -audio; it does not support -input-image, -last-frame-image, or references")
	}
	if r.Seed != 0 {
		return errors.New("lipsync does not support -seed")
	}
	return nil
}

// validateMinimaxVideo checks a request for MiniMax H3 (Replicate). H3 accepts
// text-to-video (no image), a first and/or last frame, and up to 9 reference
// images, 3 reference videos, and 3 reference audio clips. Its resolution enum
// is 768P / 2K and its ratio enum has no "auto" — image-to-video uses
// "adaptive".
func (r *Request) validateMinimaxVideo() error {
	if r.VideoDuration != 0 && (r.VideoDuration < 4 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be 4-15 seconds for MiniMax H3, got %d", r.VideoDuration)
	}
	if MinimaxVideoResolution(r.VideoResolution) == "" {
		return fmt.Errorf("video_resolution must be 768p or 2k for MiniMax H3, got %q", r.VideoResolution)
	}
	switch r.AspectRatio {
	case "adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16":
	default:
		return fmt.Errorf("MiniMax H3 aspect_ratio must be adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16; got %q", r.AspectRatio)
	}
	if len(r.InputImages) > 1 {
		return fmt.Errorf("MiniMax H3 accepts at most one -input-image first frame (use -reference-image for references), got %d", len(r.InputImages))
	}
	if r.LastFrameImage != "" && len(r.InputImages) != 1 {
		return errors.New("-last-frame-image requires exactly one -input-image first frame")
	}
	if len(r.ReferenceImages) > 9 {
		return fmt.Errorf("MiniMax H3 supports at most 9 reference images, got %d", len(r.ReferenceImages))
	}
	if len(r.ReferenceVideos) > 3 {
		return fmt.Errorf("MiniMax H3 supports at most 3 reference videos, got %d", len(r.ReferenceVideos))
	}
	if len(r.ReferenceAudios) > 3 {
		return fmt.Errorf("MiniMax H3 supports at most 3 reference audio clips, got %d", len(r.ReferenceAudios))
	}
	if r.Seed != 0 {
		return errors.New("MiniMax H3 does not support -seed")
	}
	if r.GenerateAudio != nil && !*r.GenerateAudio {
		return errors.New("MiniMax H3 has no audio toggle and does not support -no-audio")
	}
	return nil
}

// MinimaxVideoResolution maps a curds -video-resolution value onto MiniMax H3's
// resolution enum, returning "" when the value is not one H3 accepts.
func MinimaxVideoResolution(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "768p":
		return "768P"
	case "2k":
		return "2K"
	}
	return ""
}

// SeedanceLimits describes the inputs a Seedance version accepts. 2.5 raised
// every ceiling except resolution: it drops 1080p but takes far more
// references and longer clips than 2.0.
type SeedanceLimits struct {
	MaxDuration int // seconds; -1 (intelligent) is always allowed in addition
	MaxImages   int // reference images, including -input-image entries past the first
	MaxVideos   int // reference videos
	MaxAudios   int // reference audios
}

// SeedanceLimitsFor returns the input limits of the resolved Seedance model,
// falling back to 2.0's narrower contract for anything unrecognised.
func SeedanceLimitsFor(model string) SeedanceLimits {
	if IsSeedance25Model(model) {
		return SeedanceLimits{MaxDuration: 30, MaxImages: 30, MaxVideos: 10, MaxAudios: 10}
	}
	return SeedanceLimits{MaxDuration: 15, MaxImages: 9, MaxVideos: 3, MaxAudios: 3}
}

// SeedanceAllowedAspectRatios is Seedance 2.5's aspect_ratio enum. 2.0 also
// accepts "9:21" (Seedance20AspectRatios).
var SeedanceAllowedAspectRatios = map[string]bool{
	"16:9": true, "4:3": true, "1:1": true, "3:4": true,
	"9:16": true, "21:9": true, "adaptive": true,
}

// Seedance20AspectRatios is Seedance 2.0's aspect_ratio enum.
var Seedance20AspectRatios = map[string]bool{
	"16:9": true, "4:3": true, "1:1": true, "3:4": true,
	"9:16": true, "21:9": true, "9:21": true, "adaptive": true,
}

// SeedanceAspectRatiosFor returns the ratio enum of the resolved Seedance
// model.
func SeedanceAspectRatiosFor(model string) map[string]bool {
	if IsSeedance25Model(model) {
		return SeedanceAllowedAspectRatios
	}
	return Seedance20AspectRatios
}

// MaxInputImagesFor returns the cap on -input-image entries for the resolved
// model. Image models share the generic cap; the Seedance versions accept a
// first frame on top of their own model-aware reference cap, and validate
// that themselves.
func MaxInputImagesFor(model string) int {
	if IsSeedanceModel(model) {
		return 1 + SeedanceLimitsFor(model).MaxImages
	}
	return MaxInputImages
}

func (r *Request) validateSeedanceVideo() error {
	version := "2.0"
	if IsSeedance25Model(r.Model) {
		version = "2.5"
	}
	limits := SeedanceLimitsFor(r.Model)
	if r.VideoDuration != 0 && r.VideoDuration != -1 && (r.VideoDuration < 4 || r.VideoDuration > limits.MaxDuration) {
		return fmt.Errorf("video_duration must be -1 or 4-%d seconds for Seedance %s, got %d", limits.MaxDuration, version, r.VideoDuration)
	}
	// Seedance 2.5 dropped 1080p; 2.0 still offers it, so name the way out
	// instead of only rejecting the value.
	switch r.VideoResolution {
	case "480p", "720p":
	case "1080p":
		if version == "2.5" {
			return errors.New("Seedance 2.5 has no 1080p output; choose -video-resolution 720p, or -model seedance-2 / -model kling-v3 for 1080p")
		}
	case "4k":
		if version == "2.5" {
			return errors.New("Seedance 2.5 has no 4k output; choose -video-resolution 720p, or -model kling-v3 for 4k")
		}
		return fmt.Errorf("video_resolution must be 480p, 720p, or 1080p for Seedance 2.0, got %q", r.VideoResolution)
	default:
		if version == "2.5" {
			return fmt.Errorf("video_resolution must be 480p or 720p for Seedance 2.5, got %q", r.VideoResolution)
		}
		return fmt.Errorf("video_resolution must be 480p, 720p, or 1080p for Seedance 2.0, got %q", r.VideoResolution)
	}
	ratios := SeedanceAspectRatiosFor(r.Model)
	if !ratios[r.AspectRatio] {
		return fmt.Errorf("seedance aspect_ratio must be one of %s, got %q", SortedNames(ratios), r.AspectRatio)
	}
	if r.LastFrameImage != "" && len(r.InputImages) == 0 {
		return errors.New("-last-frame-image requires one -input-image first frame")
	}
	if r.LastFrameImage != "" && len(r.InputImages) != 1 {
		return errors.New("-last-frame-image requires exactly one -input-image first frame")
	}
	if len(r.InputImages) == 1 && len(r.ReferenceImages) > 0 {
		return errors.New("Seedance cannot combine first/last frame images with reference images")
	}
	referenceImageCount := len(r.ReferenceImages)
	if len(r.InputImages) > 1 {
		referenceImageCount += len(r.InputImages)
	}
	if referenceImageCount > limits.MaxImages {
		return fmt.Errorf("Seedance %s supports at most %d reference images, got %d", version, limits.MaxImages, referenceImageCount)
	}
	if len(r.ReferenceVideos) > limits.MaxVideos {
		return fmt.Errorf("Seedance %s supports at most %d reference videos, got %d", version, limits.MaxVideos, len(r.ReferenceVideos))
	}
	if len(r.ReferenceAudios) > limits.MaxAudios {
		return fmt.Errorf("Seedance %s supports at most %d reference audios, got %d", version, limits.MaxAudios, len(r.ReferenceAudios))
	}
	if len(r.ReferenceAudios) > 0 && len(r.ReferenceImages) == 0 && len(r.ReferenceVideos) == 0 && len(r.InputImages) == 0 {
		return errors.New("Seedance reference audios require at least one image or video reference")
	}
	return nil
}

// FluxAllowedAspectRatios is FLUX.2 [pro]'s ratio enum. "custom" is implied by
// -size and "match_input_image" derives the ratio from the first input image.
var FluxAllowedAspectRatios = map[string]bool{
	"match_input_image": true, "1:1": true, "16:9": true, "3:2": true,
	"2:3": true, "4:5": true, "5:4": true, "9:16": true, "3:4": true, "4:3": true,
}

// NanoBananaAllowedAspectRatios is Nano Banana 2's ratio enum.
var NanoBananaAllowedAspectRatios = map[string]bool{
	"match_input_image": true, "1:1": true, "1:4": true, "1:8": true,
	"2:3": true, "3:2": true, "3:4": true, "4:1": true, "4:3": true,
	"4:5": true, "5:4": true, "8:1": true, "9:16": true, "16:9": true, "21:9": true,
}

// FluxImageResolution maps a curds -image-resolution value onto FLUX.2's
// megapixel enum, returning "" when the value is not one FLUX accepts.
func FluxImageResolution(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "0.5mp":
		return "0.5 MP"
	case "1mp":
		return "1 MP"
	case "2mp":
		return "2 MP"
	case "4mp":
		return "4 MP"
	case "match_input_image":
		return "match_input_image"
	}
	return ""
}

// NanoBananaImageResolution maps a curds -image-resolution value onto Nano
// Banana 2's enum, returning "" when the value is not one it accepts.
func NanoBananaImageResolution(res string) string {
	switch strings.ToLower(strings.TrimSpace(res)) {
	case "1k":
		return "1K"
	case "2k":
		return "2K"
	case "4k":
		return "4K"
	}
	return ""
}

// validateFluxImage checks a request for FLUX.2 [pro] (Replicate). FLUX sizes
// output by megapixel target or by explicit width/height, produces one image
// per prediction, and has no quality/background/moderation knobs.
func (r *Request) validateFluxImage() error {
	if r.Provider != ProviderReplicate {
		return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
	}
	if r.NumImages != 1 {
		return fmt.Errorf("FLUX.2 produces one image per request, got num_images=%d", r.NumImages)
	}
	switch r.OutputFormat {
	case "webp", "png", "jpeg":
	default:
		return fmt.Errorf("FLUX.2 output_format must be webp, png, or jpeg, got %q", r.OutputFormat)
	}
	if r.Mask != "" {
		return errors.New("FLUX.2 [pro] does not accept -mask (use OpenAI edits, or flux-fill on Replicate)")
	}
	if r.Size != "" {
		w, h, ok := ParseSize(r.Size)
		if !ok {
			return fmt.Errorf("invalid -size %q; expected WxH", r.Size)
		}
		if w < 256 || w > 2048 || h < 256 || h > 2048 {
			return fmt.Errorf("FLUX.2 -size edges must be 256-2048 px, got %dx%d", w, h)
		}
	} else if !FluxAllowedAspectRatios[r.AspectRatio] {
		return fmt.Errorf("FLUX.2 aspect_ratio must be one of match_input_image, 1:1, 16:9, 3:2, 2:3, 4:5, 5:4, 9:16, 3:4, 4:3 (or pass -size WxH); got %q", r.AspectRatio)
	}
	if r.ImageResolution != "" && FluxImageResolution(r.ImageResolution) == "" {
		return fmt.Errorf("FLUX.2 -image-resolution must be 0.5mp, 1mp, 2mp, 4mp, or match_input_image; got %q", r.ImageResolution)
	}
	if len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("FLUX.2 takes reference images via -input-image, not -reference-image")
	}
	return nil
}

// validateNanoBananaImage checks a request for Google Nano Banana 2
// (Replicate). It emits jpg/png only, sizes output by 1K/2K/4K, and produces
// one image per prediction.
func (r *Request) validateNanoBananaImage() error {
	if r.Provider != ProviderReplicate {
		return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
	}
	if r.NumImages != 1 {
		return fmt.Errorf("Nano Banana 2 produces one image per request, got num_images=%d", r.NumImages)
	}
	switch r.OutputFormat {
	case "png", "jpeg":
	default:
		return fmt.Errorf("Nano Banana 2 output_format must be png or jpeg (no webp), got %q", r.OutputFormat)
	}
	if r.Mask != "" {
		return errors.New("Nano Banana 2 does not accept -mask; describe the edit in the prompt instead")
	}
	if r.Size != "" {
		return errors.New("Nano Banana 2 has no pixel-size input; use -image-resolution 1k|2k|4k with -aspect-ratio")
	}
	if !NanoBananaAllowedAspectRatios[r.AspectRatio] {
		return fmt.Errorf("Nano Banana 2 aspect_ratio must be one of match_input_image, 1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9; got %q", r.AspectRatio)
	}
	if r.ImageResolution != "" && NanoBananaImageResolution(r.ImageResolution) == "" {
		return fmt.Errorf("Nano Banana 2 -image-resolution must be 1k, 2k, or 4k; got %q", r.ImageResolution)
	}
	if r.Seed != 0 {
		return errors.New("Nano Banana 2 does not support -seed")
	}
	if len(r.ReferenceImages) > 0 || len(r.ReferenceVideos) > 0 || len(r.ReferenceAudios) > 0 {
		return errors.New("Nano Banana 2 takes reference images via -input-image, not -reference-image")
	}
	return nil
}

func (r *Request) validateSegmentation() error {
	if r.Provider != ProviderReplicate {
		return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
	}
	if len(r.InputImages) != 1 {
		return fmt.Errorf("segmentation requires exactly one -input-image, got %d", len(r.InputImages))
	}
	if r.NumImages != 1 {
		return fmt.Errorf("segmentation produces exactly one image, got num_images=%d", r.NumImages)
	}
	if r.OutputFormat != "" && r.OutputFormat != "png" {
		return fmt.Errorf("segmentation output_format must be png (transparent), got %q", r.OutputFormat)
	}
	if r.Mask != "" {
		return errors.New("segmentation does not accept -mask")
	}
	if r.Size != "" {
		return errors.New("segmentation preserves the input image size; -size is not supported")
	}
	return nil
}

func (r *Request) validateUpscale() error {
	if r.Provider != ProviderReplicate {
		return fmt.Errorf("model %q is only supported with provider replicate", r.Model)
	}
	if len(r.InputImages) != 1 {
		return fmt.Errorf("upscale requires exactly one -input-image, got %d", len(r.InputImages))
	}
	if r.NumImages != 1 {
		return fmt.Errorf("upscale produces exactly one image, got num_images=%d", r.NumImages)
	}
	switch {
	case IsPrunaUpscaleModel(r.Model):
		if r.OutputFormat != "" && !PrunaUpscaleFormats[r.OutputFormat] {
			return fmt.Errorf("upscale output_format must be png, jpeg, or webp for prunaai/p-image-upscale, got %q", r.OutputFormat)
		}
		if r.FaceEnhance {
			return errors.New("-face-enhance is not supported by prunaai/p-image-upscale; use -model upscale-pro (topazlabs/image-upscale) or -model upscale-esrgan (nightmareai/real-esrgan)")
		}
		if r.Scale != 0 && (r.Scale < 1 || r.Scale > MaxPrunaUpscaleFactor) {
			return fmt.Errorf("upscale scale must be between 1 and %d for prunaai/p-image-upscale, got %g", MaxPrunaUpscaleFactor, r.Scale)
		}
	case IsTopazUpscaleModel(r.Model):
		if r.OutputFormat != "" && r.OutputFormat != "png" {
			return fmt.Errorf("upscale output_format must be png, got %q", r.OutputFormat)
		}
		if TopazUpscaleFactor(r.Scale) == "" {
			return fmt.Errorf("topaz upscale scale must be 2, 4, or 6, got %g", r.Scale)
		}
		if r.FaceEnhance && r.Scale == 0 {
			// Topaz applies face enhancement during the upscale pass.
			return errors.New("topaz -face-enhance requires -scale 2, 4, or 6")
		}
	default:
		// nightmareai/real-esrgan: one PNG output, a numeric scale, and
		// optional GFPGAN face enhancement.
		if r.OutputFormat != "" && r.OutputFormat != "png" {
			return fmt.Errorf("upscale output_format must be png, got %q", r.OutputFormat)
		}
		if r.Scale != 0 && (r.Scale < 1 || r.Scale > MaxESRGANUpscaleFactor) {
			return fmt.Errorf("upscale scale must be between 1 and %d, got %g", MaxESRGANUpscaleFactor, r.Scale)
		}
	}
	if r.Mask != "" {
		return errors.New("upscale does not accept -mask")
	}
	if r.Size != "" {
		return errors.New("upscale derives its output size from -scale; -size is not supported")
	}
	return nil
}

// PrunaUpscaleFormats are the containers prunaai/p-image-upscale can emit, and
// the values -output-format accepts for it. curds' "jpeg" maps to its "jpg".
var PrunaUpscaleFormats = map[string]bool{"webp": true, "jpeg": true, "png": true}

// TopazUpscaleFactor maps a curds -scale value onto Topaz's upscale_factor
// enum, returning "" when the value is not one Topaz accepts. Zero means "no
// upscale, enhance only", which Topaz spells "None".
func TopazUpscaleFactor(scale float64) string {
	switch scale {
	case 0:
		return "None"
	case 2:
		return "2x"
	case 4:
		return "4x"
	case 6:
		return "6x"
	}
	return ""
}

// IsVideoModel reports whether the resolved provider model produces videos.
func IsVideoModel(model string) bool {
	return IsSeedanceModel(model) || IsGrokImagineVideoModel(model) ||
		IsXaiVideoModel(model) || IsMinimaxVideoModel(model) ||
		IsKlingVideoModel(model) || IsKlingAvatarModel(model) ||
		IsLipsyncModel(model)
}

// IsMinimaxVideoModel reports whether the resolved provider model is MiniMax's
// H3 multimodal video model, served by Replicate.
func IsMinimaxVideoModel(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	switch model {
	case MinimaxVideoModel:
		return true
	}
	return strings.HasPrefix(model, MinimaxVideoModel+":")
}

// IsXaiVideoModel reports whether the resolved model is xAI's native
// Grok Imagine Video model, served by the x.ai API (provider "xai"). This is
// distinct from the Replicate-hosted "xai/grok-imagine-video-1.5" wrapper.
func IsXaiVideoModel(model string) bool {
	return strings.TrimSpace(strings.ToLower(model)) == DefaultXaiVideoModel
}

// IsSeedanceModel reports whether the resolved provider model is any ByteDance
// Seedance video model (2.0 or 2.5). Version-specific behaviour goes through
// IsSeedance25Model / SeedanceLimitsFor.
func IsSeedanceModel(model string) bool {
	return IsSeedance25Model(model) || IsSeedance20Model(model)
}

// IsSeedance25Model reports whether the resolved provider model is Seedance
// 2.5, the default video model (480p/720p, up to 30 reference images).
func IsSeedance25Model(model string) bool {
	return matchesReplicateModel(model, Seedance25VideoModel)
}

// IsSeedance20Model reports whether the resolved provider model is Seedance
// 2.0 (or its fast variant) — the version that still offers 1080p.
func IsSeedance20Model(model string) bool {
	return matchesReplicateModel(model, SeedanceVideoModel) ||
		matchesReplicateModel(model, SeedanceVideoModel+"-fast")
}

// IsGrokImagineVideoModel reports whether the resolved provider model is xAI's
// Grok Imagine Video image-to-video model.
func IsGrokImagineVideoModel(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	switch model {
	case GrokImagineVideoModel:
		return true
	}
	return strings.HasPrefix(model, GrokImagineVideoModel+":")
}

// IsKlingVideoModel reports whether the resolved provider model is Kling Video
// 3.0, served by Replicate.
func IsKlingVideoModel(model string) bool {
	return matchesReplicateModel(model, KlingVideoModel)
}

// IsKlingAvatarModel reports whether the resolved provider model is Kling
// Avatar 2.0 (a talking head built from one portrait plus one audio clip),
// served by Replicate.
func IsKlingAvatarModel(model string) bool {
	return matchesReplicateModel(model, KlingAvatarModel)
}

// IsLipsyncModel reports whether the resolved provider model is Sync Labs'
// lipsync-2-pro, served by Replicate.
func IsLipsyncModel(model string) bool {
	return matchesReplicateModel(model, LipsyncModel)
}

// IsFluxImageModel reports whether the resolved provider model is FLUX.2 [pro].
func IsFluxImageModel(model string) bool {
	return matchesReplicateModel(model, FluxImageModel)
}

// IsNanoBananaImageModel reports whether the resolved provider model is Google
// Nano Banana 2.
func IsNanoBananaImageModel(model string) bool {
	return matchesReplicateModel(model, NanoBananaImageModel)
}

// IsTopazUpscaleModel reports whether the resolved provider model is Topaz
// Labs' image upscaler (-model upscale-pro).
func IsTopazUpscaleModel(model string) bool {
	return matchesReplicateModel(model, TopazUpscaleModel)
}

// IsPrunaUpscaleModel reports whether the resolved provider model is
// prunaai/p-image-upscale, the default upscaler (-model upscale).
func IsPrunaUpscaleModel(model string) bool {
	return matchesReplicateModel(model, DefaultUpscaleModel)
}

// IsRealESRGANUpscaleModel reports whether the resolved provider model is
// nightmareai/real-esrgan (-model upscale-esrgan), the one upscaler with face
// enhancement.
func IsRealESRGANUpscaleModel(model string) bool {
	return matchesReplicateModel(model, RealESRGANUpscaleModel)
}

// IsMusicModel reports whether the resolved provider model is ElevenLabs Music
// (elevenlabs/music), the default music model: prompt-in, instrumental by
// default, exact requested length.
func IsMusicModel(model string) bool {
	return matchesReplicateModel(model, MusicModel)
}

// IsMusicVocalModel reports whether the resolved provider model is MiniMax
// Music 2.6 (minimax/music-2.6): a full song, with vocals by default, from a
// prompt and/or -lyrics.
func IsMusicVocalModel(model string) bool {
	return matchesReplicateModel(model, MusicVocalModel)
}

// IsSFXModel reports whether the resolved provider model is Stable Audio 2.5
// (stability-ai/stable-audio-2.5): sound effects, ambience, and short cues.
func IsSFXModel(model string) bool {
	return matchesReplicateModel(model, SFXModel)
}

// IsTTSSpeechModel reports whether the resolved provider model is MiniMax
// Speech 2.8 HD (minimax/speech-2.8-hd): narration with a free-form voice id,
// emotion, speed, and pitch (-model tts-minimax).
func IsTTSSpeechModel(model string) bool {
	return matchesReplicateModel(model, TTSSpeechModel)
}

// IsTTSGeminiModel reports whether the resolved provider model is Google's
// Gemini 3.1 Flash TTS (google/gemini-3.1-flash-tts), the default TTS model:
// an enum voice plus a natural-language style prompt (-model tts).
func IsTTSGeminiModel(model string) bool {
	return matchesReplicateModel(model, TTSGeminiModel)
}

// IsTTSElevenLabsModel reports whether the resolved provider model is
// ElevenLabs v3 (elevenlabs/v3): expressive text-to-speech with
// stability/style, served by Replicate.
func IsTTSElevenLabsModel(model string) bool {
	return matchesReplicateModel(model, TTSElevenLabsModel)
}

// IsOpenAITTSMiniModel reports whether the resolved provider model is OpenAI's
// gpt-4o-mini-tts, the OpenAI speech model that accepts -instructions.
func IsOpenAITTSMiniModel(model string) bool {
	return matchesReplicateModel(model, TTSOpenAIModel)
}

// IsOpenAITTSModel reports whether the resolved provider model is one of
// OpenAI's speech models (gpt-4o-mini-tts or tts-1-hd), served by the openai
// provider's POST /v1/audio/speech endpoint.
func IsOpenAITTSModel(model string) bool {
	return IsOpenAITTSMiniModel(model) || matchesReplicateModel(model, TTS1HDModel)
}

// IsTTSModel reports whether the resolved provider model is a text-to-speech
// model: Gemini 3.1 Flash TTS, MiniMax Speech 2.8 HD, ElevenLabs v3, or an
// OpenAI speech model.
func IsTTSModel(model string) bool {
	return IsTTSGeminiModel(model) || IsTTSSpeechModel(model) ||
		IsTTSElevenLabsModel(model) || IsOpenAITTSModel(model)
}

// IsAudioModel reports whether the resolved provider model produces audio
// rather than an image or a video. Audio models are prompt-driven but run
// under their own validation and request-building path: no input media, no
// aspect ratio, no pixel size, and an mp3/wav output container. It covers the
// music/sfx models and the text-to-speech models.
func IsAudioModel(model string) bool {
	return IsMusicModel(model) || IsMusicVocalModel(model) || IsSFXModel(model) || IsTTSModel(model)
}

// matchesReplicateModel reports whether model is base, optionally with a
// ":version" pin. Case- and whitespace-insensitive. Never substring-match a
// model name: "black-forest-labs/flux-2-pro-evil/x" must not match.
func matchesReplicateModel(model, base string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	return model == base || strings.HasPrefix(model, base+":")
}

// IsSegmentationModel reports whether the resolved provider model performs
// image segmentation / background removal. These models take an input image
// and return a transparent PNG instead of generating new pixels from a
// prompt, so they need a different validation and request-building path.
func IsSegmentationModel(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	switch model {
	case "bria/remove-background":
		return true
	}
	return strings.HasPrefix(model, "bria/remove-background:")
}

// IsUpscaleModel reports whether the resolved provider model performs
// super-resolution / upscaling. Like segmentation models these take an input
// image (plus a scale factor) and return a single image rather than generating
// new pixels from a prompt, so they share the no-prompt validation and
// request-building path.
func IsUpscaleModel(model string) bool {
	return IsPrunaUpscaleModel(model) || IsRealESRGANUpscaleModel(model) || IsTopazUpscaleModel(model)
}

// IsPromptlessModel reports whether the model runs without a text prompt: it is
// driven by input media (segmentation, upscaling, lip-sync) or treats the
// prompt as optional (Kling Avatar 2.0).
func IsPromptlessModel(model string) bool {
	return IsSegmentationModel(model) || IsUpscaleModel(model) ||
		IsLipsyncModel(model) || IsKlingAvatarModel(model)
}

// AutoDetectProvider picks a provider based on the supplied env lookup.
// OpenAI is preferred when both keys are present.
func AutoDetectProvider(getenv func(string) string) string {
	if getenv == nil {
		return ""
	}
	if getenv("OPENAI_API_KEY") != "" {
		return ProviderOpenAI
	}
	if getenv("REPLICATE_API_TOKEN") != "" {
		return ProviderReplicate
	}
	return ""
}

// gpt-image-2 size constraints, per OpenAI docs.
const (
	EdgeMultiple   = 16      // both edges must be multiples of this
	MaxEdge        = 3840    // longest edge cap
	MinTotalPixels = 655_360 // also enforced by upstream
	MaxTotalPixels = 8_294_400
	MaxRatio       = 3.0 // long edge / short edge
)

// ResolveSize returns the size string to send upstream. If req.Size is set,
// it's parsed and rounded to satisfy gpt-image-2's constraints (multiples of
// 16, max edge 3840, max 3:1 ratio). When Size is empty, AspectRatio is
// looked up in AspectRatioSizes. Anything unparseable falls back to "auto".
func ResolveSize(req *Request) string {
	if req.Size != "" {
		if strings.EqualFold(req.Size, "auto") {
			return "auto"
		}
		if w, h, ok := ParseSize(req.Size); ok {
			rw, rh, _ := RoundSize(w, h)
			return fmt.Sprintf("%dx%d", rw, rh)
		}
		return req.Size
	}
	if s, ok := AspectRatioSizes[req.AspectRatio]; ok {
		return s
	}
	return "auto"
}

// ParseSize parses "WxH" (case-insensitive). Returns (w, h, ok).
func ParseSize(s string) (int, int, bool) {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(s)), "x")
	if len(parts) != 2 {
		return 0, 0, false
	}
	w, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
	h, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err1 != nil || err2 != nil || w <= 0 || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}

// RoundSize nudges (w,h) into a gpt-image-2-valid pair: both edges become
// multiples of EdgeMultiple, neither exceeds MaxEdge, and the long-to-short
// ratio is clamped to MaxRatio. Returns (w, h, changed).
func RoundSize(w, h int) (int, int, bool) {
	rw := nearestMultiple(w, EdgeMultiple)
	rh := nearestMultiple(h, EdgeMultiple)

	if rw > MaxEdge {
		rw = MaxEdge
	}
	if rh > MaxEdge {
		rh = MaxEdge
	}
	if rw < EdgeMultiple {
		rw = EdgeMultiple
	}
	if rh < EdgeMultiple {
		rh = EdgeMultiple
	}

	// Clamp ratio to 3:1.
	if float64(rw)/float64(rh) > MaxRatio {
		rw = nearestMultiple(int(float64(rh)*MaxRatio), EdgeMultiple)
	}
	if float64(rh)/float64(rw) > MaxRatio {
		rh = nearestMultiple(int(float64(rw)*MaxRatio), EdgeMultiple)
	}
	return rw, rh, rw != w || rh != h
}

func nearestMultiple(v, base int) int {
	if v <= 0 {
		return base
	}
	rem := v % base
	if rem == 0 {
		return v
	}
	if rem*2 >= base {
		return v + (base - rem)
	}
	return v - rem
}

// SortedAspectRatios returns a deterministic, human-friendly list of
// supported ratios.
func SortedAspectRatios() []string {
	keys := make([]string, 0, len(AspectRatioSizes))
	for k := range AspectRatioSizes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func detectMime(path string, data []byte) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	case ".gif":
		return "image/gif"
	case ".mp4":
		return "video/mp4"
	case ".mov":
		return "video/quicktime"
	case ".mp3":
		return "audio/mpeg"
	case ".wav":
		return "audio/wav"
	case ".m4a":
		return "audio/mp4"
	}
	return http.DetectContentType(data)
}

func httpClientOrDefault(c *http.Client) *http.Client {
	if c == nil {
		return http.DefaultClient
	}
	return c
}
