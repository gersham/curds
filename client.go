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
)

const (
	ProviderReplicate = "replicate"
	ProviderOpenAI    = "openai"
	ProviderXai       = "xai"

	DefaultReplicateModel = "openai/gpt-image-2"
	DefaultOpenAIModel    = GPTImage25Model

	// GPTImage25Model is the gpt-image-2.5 model curds sends to the OpenAI
	// Image API and the default image model. It takes the same request
	// surface as gpt-image-2 on /v1/images/generations and /v1/images/edits
	// (model, prompt, n, size, quality, background, moderation,
	// output_format, output_compression, user), so nothing about sizing,
	// aspect ratios, or output formats changes with it — only the id.
	GPTImage25Model = "gpt-image-2.5-flare"

	// GPTImage2Model is the previous-generation image model, still
	// selectable via -model gpt-image-2.
	GPTImage2Model = "gpt-image-2"

	// GrokImagineVideoModel is the Replicate-hosted Grok Imagine Video 1.5
	// wrapper (image-to-video only) and the default Replicate video model.
	GrokImagineVideoModel = "xai/grok-imagine-video-1.5"

	// DefaultVideoModel is the Replicate-hosted video model used when output is
	// MP4 and no model is supplied. Seedance 2.0 leads on reference handling,
	// multi-scene continuity, and native audio.
	DefaultVideoModel = SeedanceVideoModel

	// MinimaxVideoModel is MiniMax H3 on Replicate: multimodal text-to-video,
	// first/last-frame image-to-video, and reference images/videos/audio.
	// Selectable via -model minimax-h3; not a default.
	MinimaxVideoModel = "minimax/h3"

	// SeedanceVideoModel is ByteDance Seedance 2.0 on Replicate and the default
	// video model: text-to-video, first/last frame, reference images/videos/
	// audio, and native synchronized audio in the same pass.
	SeedanceVideoModel = "bytedance/seedance-2.0"

	// KlingVideoModel is Kling Video 3.0 on Replicate: text-to-video and
	// image-to-video with start/end frames, native audio with lip-synced
	// dialogue, and standard (720p) / pro (1080p) / 4k modes.
	KlingVideoModel = "kwaivgi/kling-v3-video"

	// FluxImageModel is FLUX.2 [pro] on Replicate: fast, cheap image generation
	// with reference-image control. Uses megapixel resolutions, not sizes.
	FluxImageModel = "black-forest-labs/flux-2-pro"

	// NanoBananaImageModel is Google Nano Banana 2 on Replicate: strong
	// character consistency and multi-image compositing, 1K/2K/4K output.
	NanoBananaImageModel = "google/nano-banana-2"

	// TopazUpscaleModel is Topaz Labs' image upscaler on Replicate: a modern
	// alternative to Real-ESRGAN with 2x/4x/6x factors and face enhancement.
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
	// when -model upscale is requested. nightmareai/real-esrgan takes a single
	// `image` URL plus a `scale` factor and optional `face_enhance`, returning
	// one upscaled PNG. It generates no new pixels from a prompt, so it shares
	// the segmentation request/validation path.
	DefaultUpscaleModel = "nightmareai/real-esrgan"

	// DefaultUpscaleScale is the default super-resolution factor sent when the
	// caller leaves Request.Scale at zero. Matches real-esrgan's own default.
	DefaultUpscaleScale = 4

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
	Quality           string // low, medium, high, auto
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
	FaceEnhance       bool    // run GFPGAN face enhancement (upscale models only)

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

// Result groups all assets produced by a Request.
type Result struct {
	Images []Image
	Videos []Video
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
		case IsMinimaxVideoModel(r.Model):
			// H3's vocabulary is 768P / 2K; default to the cheaper tier.
			r.VideoResolution = "768p"
		case IsKlingVideoModel(r.Model):
			// Matches Kling's own default mode ("pro").
			r.VideoResolution = "1080p"
		default:
			r.VideoResolution = "720p"
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
	if !IsSegmentationModel(r.Model) && !IsUpscaleModel(r.Model) && strings.TrimSpace(r.Prompt) == "" {
		return errors.New("prompt is required")
	}
	if r.NumImages < 1 || r.NumImages > 10 {
		return fmt.Errorf("num_images must be 1-10, got %d", r.NumImages)
	}
	if r.OutputCompression < 0 || r.OutputCompression > 100 {
		return fmt.Errorf("output_compression must be 0-100, got %d", r.OutputCompression)
	}
	if len(r.InputImages) > MaxInputImages {
		return fmt.Errorf("at most %d input images supported, got %d", MaxInputImages, len(r.InputImages))
	}
	switch {
	case IsVideoModel(r.Model):
		if err := r.validateVideo(); err != nil {
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
	if r.Provider == ProviderReplicate && !IsVideoModel(r.Model) && !IsSegmentationModel(r.Model) &&
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
	default:
		return fmt.Errorf("unsupported video model %q", r.Model)
	}
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

func (r *Request) validateSeedanceVideo() error {
	if r.VideoDuration != 0 && r.VideoDuration != -1 && (r.VideoDuration < 4 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be -1 or 4-15 seconds, got %d", r.VideoDuration)
	}
	switch r.VideoResolution {
	case "480p", "720p", "1080p":
	default:
		return fmt.Errorf("video_resolution must be 480p, 720p, or 1080p, got %q", r.VideoResolution)
	}
	switch r.AspectRatio {
	case "16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "9:21", "adaptive":
	default:
		return fmt.Errorf("seedance aspect_ratio must be 16:9, 4:3, 1:1, 3:4, 9:16, 21:9, 9:21, or adaptive; got %q", r.AspectRatio)
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
	if referenceImageCount > 9 {
		return fmt.Errorf("Seedance supports at most 9 reference images, got %d", referenceImageCount)
	}
	if len(r.ReferenceVideos) > 3 {
		return fmt.Errorf("Seedance supports at most 3 reference videos, got %d", len(r.ReferenceVideos))
	}
	if len(r.ReferenceAudios) > 3 {
		return fmt.Errorf("Seedance supports at most 3 reference audios, got %d", len(r.ReferenceAudios))
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
	if r.OutputFormat != "" && r.OutputFormat != "png" {
		return fmt.Errorf("upscale output_format must be png, got %q", r.OutputFormat)
	}
	if IsTopazUpscaleModel(r.Model) {
		if TopazUpscaleFactor(r.Scale) == "" {
			return fmt.Errorf("topaz upscale scale must be 2, 4, or 6, got %g", r.Scale)
		}
		if r.FaceEnhance && r.Scale == 0 {
			// Topaz applies face enhancement during the upscale pass.
			return errors.New("topaz -face-enhance requires -scale 2, 4, or 6")
		}
	} else if r.Scale != 0 && (r.Scale < 1 || r.Scale > 10) {
		return fmt.Errorf("upscale scale must be between 1 and 10, got %g", r.Scale)
	}
	if r.Mask != "" {
		return errors.New("upscale does not accept -mask")
	}
	if r.Size != "" {
		return errors.New("upscale derives its output size from -scale; -size is not supported")
	}
	return nil
}

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
		IsKlingVideoModel(model)
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

// IsSeedanceModel reports whether the resolved provider model is ByteDance's
// Seedance video model.
func IsSeedanceModel(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	switch model {
	case "bytedance/seedance-2.0", "bytedance/seedance-2.0-fast":
		return true
	}
	return strings.HasPrefix(model, "bytedance/seedance-2.0:") ||
		strings.HasPrefix(model, "bytedance/seedance-2.0-fast:")
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
// Labs' image upscaler.
func IsTopazUpscaleModel(model string) bool {
	return matchesReplicateModel(model, TopazUpscaleModel)
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
	return matchesReplicateModel(model, DefaultUpscaleModel) || IsTopazUpscaleModel(model)
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
