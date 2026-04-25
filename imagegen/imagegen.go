// Package imagegen generates images via image-generation providers.
//
// Supported providers:
//   - openai    (direct OpenAI Image API; default model gpt-image-2)
//   - replicate (Replicate-hosted models; default openai/gpt-image-2)
//
// The package is transport-agnostic and intended to be reusable from a CLI,
// HTTP service, or background worker.
package imagegen

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	ProviderReplicate = "replicate"
	ProviderOpenAI    = "openai"

	DefaultReplicateModel = "openai/gpt-image-2"
	DefaultOpenAIModel    = "gpt-image-2"

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

// Request describes a single image-generation call.
type Request struct {
	Provider          string
	Token             string
	Model             string // empty = provider default
	Prompt            string
	AspectRatio       string // e.g. "16:9"; ignored if Size is set
	Size              string // e.g. "2048x1152"; OpenAI only
	Quality           string // low, medium, high, auto
	NumImages         int
	OutputFormat      string // webp, png, jpeg
	OutputCompression int    // 0-100; OpenAI webp/jpeg only
	Background        string // auto, opaque
	Moderation        string // auto, low
	User              string // OpenAI only
	ReplicateBYOKey   string // optional OpenAI key passed through Replicate
	InputImages       []string
	Mask              string // OpenAI edits only

	PollInterval time.Duration // Replicate poll cadence; 0 = default
	Logger       io.Writer     // optional verbose logging sink
}

// Image is one rendered image.
type Image struct {
	Bytes         []byte
	Format        string
	RevisedPrompt string // populated for OpenAI when available
}

// Result groups all images produced by a Request.
type Result struct {
	Images []Image
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
}

// New constructs a Client with default providers wired to the public APIs.
func New() *Client {
	hc := &http.Client{Timeout: 0}
	return &Client{
		HTTPClient: hc,
		Replicate:  &ReplicateProvider{HTTPClient: hc, APIBase: "https://api.replicate.com/v1"},
		OpenAI:     &OpenAIProvider{HTTPClient: hc, APIBase: "https://api.openai.com/v1"},
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
		r.AspectRatio = "1:1"
	}
	if r.Quality == "" {
		r.Quality = "auto"
	}
	if r.OutputFormat == "" {
		r.OutputFormat = "webp"
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
	case ProviderReplicate, ProviderOpenAI:
	default:
		return fmt.Errorf("unsupported provider %q (supported: openai, replicate)", r.Provider)
	}
	if r.Token == "" {
		return fmt.Errorf("missing %s token", r.Provider)
	}
	if strings.TrimSpace(r.Prompt) == "" {
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
	switch r.OutputFormat {
	case "webp", "png", "jpeg":
	default:
		return fmt.Errorf("output_format must be webp, png, or jpeg, got %q", r.OutputFormat)
	}
	if r.Provider == ProviderReplicate && r.AspectRatio != "" && !ReplicateAllowedAspectRatios[r.AspectRatio] {
		return fmt.Errorf("replicate only supports 1:1, 3:2, 2:3 aspect ratios; got %q", r.AspectRatio)
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
	}
	return ""
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

// ResolveSize returns the explicit size or the size mapped from the aspect
// ratio. Falls back to "auto" when neither matches a known ratio.
func ResolveSize(req *Request) string {
	if req.Size != "" {
		return req.Size
	}
	if s, ok := AspectRatioSizes[req.AspectRatio]; ok {
		return s
	}
	return "auto"
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

func logf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, format, args...)
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
	}
	return http.DetectContentType(data)
}

func httpClientOrDefault(c *http.Client) *http.Client {
	if c == nil {
		return http.DefaultClient
	}
	return c
}
