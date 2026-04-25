// Command replicate-image-gen is a thin CLI over the imagegen library.
//
// All HTTP/provider logic lives in github.com/gersham/replicate-image-gen/imagegen
// so this binary stays a translation layer between flags and library calls.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gersham/replicate-image-gen/imagegen"
)

const defaultOutput = "/tmp/replicate-generated-image.webp"

// imageList accepts both repeated flags and comma-separated values.
type imageList []string

func (s *imageList) String() string { return strings.Join(*s, ",") }
func (s *imageList) Set(v string) error {
	for _, p := range strings.Split(v, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			*s = append(*s, p)
		}
	}
	return nil
}

type cliOptions struct {
	provider          string
	token             string
	model             string
	prompt            string
	outputPath        string
	aspectRatio       string
	size              string
	quality           string
	numImages         int
	outputFormat      string
	outputCompression int
	background        string
	moderation        string
	user              string
	replicateBYOKey   string
	inputImages       imageList
	mask              string
	pollInterval      time.Duration
	timeout           time.Duration
	verbose           bool
}

func main() {
	opts, err := parseFlags()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		fmt.Fprintln(os.Stderr, "")
		flag.Usage()
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if opts.timeout > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, opts.timeout)
		defer stop()
	}

	if err := run(ctx, opts); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func setupUsage() {
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		ratios := joinRatios(imagegen.AspectRatioSizes)
		fmt.Fprintf(out, `replicate-image-gen — generate images via OpenAI or Replicate

USAGE
  replicate-image-gen [flags]
  echo "your prompt" | replicate-image-gen [flags]

PROVIDERS
  Auto-selected from environment when -provider is omitted (OpenAI preferred):
    OPENAI_API_KEY       → -provider openai (direct, native gpt-image-2)
    REPLICATE_API_TOKEN  → -provider replicate (Replicate hosted)
  Override with -provider openai|replicate.

ASPECT RATIOS (for -aspect-ratio)
  Replicate gpt-image-2: 1:1, 3:2, 2:3
  OpenAI gpt-image-2:    %s
  Or pass -size WxH (multiples of 16, max edge 3840) to override.

EXAMPLES
  # Auto-detected provider
  replicate-image-gen -p "a watercolor of a fox in a meadow"

  # OpenAI direct, ~1080p landscape, save somewhere specific
  replicate-image-gen -provider openai -aspect-ratio 16:9 -p "neon city" -o /tmp/city.webp

  # Edit / compose with reference images (comma-separated, up to %d)
  replicate-image-gen -i a.png,b.png,c.png -p "compose them"

  # Long prompt from stdin
  cat prompt.txt | replicate-image-gen -aspect-ratio 9:16 -quality high

FLAGS
`, ratios, imagegen.MaxInputImages)
		flag.PrintDefaults()
	}
}

func joinRatios(m map[string]string) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s (%s)", k, m[k]))
	}
	return strings.Join(parts, ", ")
}

func parseFlags() (*cliOptions, error) {
	opts := &cliOptions{}

	flag.StringVar(&opts.provider, "provider", "", "Provider: openai, replicate (default: auto-detect from env, openai preferred)")
	flag.StringVar(&opts.token, "token", "", "Provider API token (default: $OPENAI_API_KEY or $REPLICATE_API_TOKEN)")
	flag.StringVar(&opts.token, "t", "", "alias for -token")
	flag.StringVar(&opts.model, "model", "", "Model name (defaults: openai=gpt-image-2, replicate=openai/gpt-image-2)")
	flag.StringVar(&opts.model, "m", "", "alias for -model")
	flag.StringVar(&opts.prompt, "prompt", "", "Prompt text (reads stdin if omitted)")
	flag.StringVar(&opts.prompt, "p", "", "alias for -prompt")
	flag.StringVar(&opts.outputPath, "output", defaultOutput, "Output file path. With multiple images, suffix -1, -2, ... is inserted before the extension.")
	flag.StringVar(&opts.outputPath, "o", defaultOutput, "alias for -output")

	flag.StringVar(&opts.aspectRatio, "aspect-ratio", "1:1", "Aspect ratio (see ASPECT RATIOS section). Replicate restricts to 1:1/3:2/2:3.")
	flag.StringVar(&opts.size, "size", "", "Explicit pixel size (openai), e.g. 2048x1152 or auto. Overrides -aspect-ratio.")
	flag.StringVar(&opts.quality, "quality", "auto", "Quality: low, medium, high, auto")
	flag.IntVar(&opts.numImages, "number-of-images", 1, "Number of images (1-10)")
	flag.StringVar(&opts.outputFormat, "output-format", "webp", "Output format: webp, png, jpeg")
	flag.IntVar(&opts.outputCompression, "output-compression", 90, "Output compression 0-100 (openai webp/jpeg only)")
	flag.StringVar(&opts.background, "background", "auto", "Background: auto, opaque (gpt-image-2 does not support transparent)")
	flag.StringVar(&opts.moderation, "moderation", "auto", "Moderation: auto, low")
	flag.StringVar(&opts.user, "user", "", "End-user identifier (openai only)")

	flag.StringVar(&opts.replicateBYOKey, "replicate-openai-api-key", "", "BYO OpenAI key passed through to Replicate (replicate provider only)")

	flag.Var(&opts.inputImages, "input-image", fmt.Sprintf("Input reference image(s); repeat or comma-separate, up to %d (file path, http(s) URL, or data: URL)", imagegen.MaxInputImages))
	flag.Var(&opts.inputImages, "i", "alias for -input-image")
	flag.StringVar(&opts.mask, "mask", "", "Mask image file (openai edits only)")

	flag.DurationVar(&opts.pollInterval, "poll-interval", 2*time.Second, "Polling interval while prediction runs (replicate)")
	flag.DurationVar(&opts.timeout, "timeout", 10*time.Minute, "Overall timeout (0 disables)")
	flag.BoolVar(&opts.verbose, "verbose", false, "Verbose logging to stderr")
	flag.BoolVar(&opts.verbose, "v", false, "alias for -verbose")

	setupUsage()
	flag.Parse()

	opts.provider = strings.ToLower(strings.TrimSpace(opts.provider))
	if opts.provider == "" || opts.provider == "auto" {
		opts.provider = imagegen.AutoDetectProvider(os.Getenv)
		if opts.provider == "" {
			return nil, errors.New("no provider token in env. Set OPENAI_API_KEY or REPLICATE_API_TOKEN, or pass -provider/-token")
		}
	}

	if opts.token == "" {
		switch opts.provider {
		case imagegen.ProviderReplicate:
			opts.token = os.Getenv("REPLICATE_API_TOKEN")
		case imagegen.ProviderOpenAI:
			opts.token = os.Getenv("OPENAI_API_KEY")
		}
	}

	if opts.prompt == "" {
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				return nil, fmt.Errorf("read stdin: %w", err)
			}
			opts.prompt = strings.TrimSpace(string(b))
		}
	}

	return opts, nil
}

func run(ctx context.Context, opts *cliOptions) error {
	req := &imagegen.Request{
		Provider:          opts.provider,
		Token:             opts.token,
		Model:             opts.model,
		Prompt:            opts.prompt,
		AspectRatio:       opts.aspectRatio,
		Size:              opts.size,
		Quality:           opts.quality,
		NumImages:         opts.numImages,
		OutputFormat:      opts.outputFormat,
		OutputCompression: opts.outputCompression,
		Background:        opts.background,
		Moderation:        opts.moderation,
		User:              opts.user,
		ReplicateBYOKey:   opts.replicateBYOKey,
		InputImages:       []string(opts.inputImages),
		Mask:              opts.mask,
		PollInterval:      opts.pollInterval,
	}
	if opts.verbose {
		req.Logger = os.Stderr
		fmt.Fprintf(os.Stderr, "provider=%s\n", req.Provider)
	}

	res, err := imagegen.New().Generate(ctx, req)
	if err != nil {
		return err
	}
	if res == nil || len(res.Images) == 0 {
		return errors.New("no images returned")
	}
	return saveImages(opts, res.Images)
}

func saveImages(opts *cliOptions, images []imagegen.Image) error {
	ext := filepath.Ext(opts.outputPath)
	stem := strings.TrimSuffix(opts.outputPath, ext)
	if ext == "" {
		ext = "." + opts.outputFormat
	}

	for i, img := range images {
		path := opts.outputPath
		if len(images) > 1 {
			path = fmt.Sprintf("%s-%d%s", stem, i+1, ext)
		}
		if dir := filepath.Dir(path); dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		if err := os.WriteFile(path, img.Bytes, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", path, err)
		}
		if opts.verbose {
			fmt.Fprintf(os.Stderr, "wrote %s (%d bytes)\n", path, len(img.Bytes))
			if img.RevisedPrompt != "" {
				fmt.Fprintf(os.Stderr, "revised_prompt[%d]: %s\n", i, img.RevisedPrompt)
			}
		}
		fmt.Println(path)
	}
	return nil
}
