// Package curds generates images and videos via generation providers.
//
// Supported providers:
//   - openai    (direct OpenAI Image API; default model gpt-image-2)
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
	DefaultOpenAIModel    = "gpt-image-2"
	DefaultVideoModel     = "xai/grok-imagine-video-1.5"

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
		if IsGrokImagineVideoModel(r.Model) || IsXaiVideoModel(r.Model) {
			r.AspectRatio = "auto"
		} else {
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
		default:
			r.OutputFormat = "webp"
		}
	}
	if IsVideoModel(r.Model) && r.VideoResolution == "" {
		r.VideoResolution = "720p"
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
	default:
		switch r.OutputFormat {
		case "webp", "png", "jpeg":
		default:
			return fmt.Errorf("output_format must be webp, png, or jpeg, got %q", r.OutputFormat)
		}
	}
	if r.Provider == ProviderReplicate && !IsVideoModel(r.Model) && !IsSegmentationModel(r.Model) && !IsUpscaleModel(r.Model) && r.AspectRatio != "" && !ReplicateAllowedAspectRatios[r.AspectRatio] {
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
	default:
		return fmt.Errorf("unsupported video model %q", r.Model)
	}
}

// validateXaiVideo checks a request for xAI's native Grok Imagine Video API.
// Unlike the Replicate wrapper this supports text-to-video (image optional),
// 1080p, and reference images. The source image goes via -input-image (0 or 1)
// and additional references via -reference-image.
func (r *Request) validateXaiVideo() error {
	if len(r.InputImages) > 1 {
		return fmt.Errorf("xai video accepts at most one -input-image source (use -reference-image for additional references), got %d", len(r.InputImages))
	}
	if r.VideoDuration != 0 && (r.VideoDuration < 1 || r.VideoDuration > 15) {
		return fmt.Errorf("video_duration must be 1-15 seconds for grok-imagine-video, got %d", r.VideoDuration)
	}
	switch r.VideoResolution {
	case "480p", "720p", "1080p":
	default:
		return fmt.Errorf("video_resolution must be 480p, 720p, or 1080p for grok-imagine-video, got %q", r.VideoResolution)
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
	if r.Scale != 0 && (r.Scale < 1 || r.Scale > 10) {
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

// DefaultModel returns the provider's default model name.
func DefaultModel(provider string) string {
	switch provider {
	case ProviderReplicate:
		return DefaultReplicateModel
	case ProviderOpenAI:
		return DefaultOpenAIModel
	case ProviderXai:
		return DefaultXaiVideoModel
	}
	return ""
}

// IsVideoModel reports whether the resolved provider model produces videos.
func IsVideoModel(model string) bool {
	return IsSeedanceModel(model) || IsGrokImagineVideoModel(model) || IsXaiVideoModel(model)
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
	case DefaultVideoModel:
		return true
	}
	return strings.HasPrefix(model, DefaultVideoModel+":")
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
	model = strings.TrimSpace(strings.ToLower(model))
	switch model {
	case "nightmareai/real-esrgan":
		return true
	}
	return strings.HasPrefix(model, "nightmareai/real-esrgan:")
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
