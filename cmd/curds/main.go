// Command curds generates images via OpenAI or Replicate.
//
// Build:
//
//	go build -o curds .
//
// On first run, curds writes a default config at ~/.config/curds/config.toml.
// Token resolution priority: config file > .env in cwd > process env vars.
// When the prompt or required token is missing, curds drops into a small
// interactive TUI driven by charmbracelet/huh.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/gersham/curds"
	"github.com/gersham/curds/config"
	"github.com/gersham/curds/tui"
)

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
	tokenFlag         string
	modelKey          string // user-facing -model arg (a config key OR raw model name)
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
	noTUI             bool
	open              bool
	inline            string // "auto" | "on" | "off"
}

func main() {
	logger := newLogger(os.Stderr)
	start := time.Now()

	if err := realMain(logger, start); err != nil {
		logger.error("curds.failed", "err", err.Error(), "duration_ms", time.Since(start).Milliseconds())
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func realMain(logger *logfmtLogger, start time.Time) error {
	logger.info("curds.start", "version", "0.1.0")

	opts, err := parseFlags()
	if err != nil {
		flag.Usage()
		return err
	}

	cfg, created, err := config.LoadOrCreate()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	if created {
		logger.info("config.created", "path", cfg.Path)
		fmt.Fprintf(os.Stderr, "wrote default config to %s\n", cfg.Path)
	}
	logger.info("config.loaded", "path", cfg.Path)

	dotenv, err := config.LoadDotEnv(".env")
	if err != nil {
		logger.error("dotenv.read_failed", "err", err.Error())
	} else if dotenv != nil {
		logger.info("dotenv.loaded", "keys", len(dotenv))
	}

	applyConfigDefaults(opts, cfg)

	// Resolve provider.
	if opts.provider == "" {
		opts.provider = config.DetectProvider(cfg, dotenv, os.Getenv)
	}

	// Resolve token if user didn't pass -token. CLI flag overrides everything.
	token := opts.tokenFlag
	if token == "" && opts.provider != "" {
		token = config.ResolveToken(opts.provider, cfg, dotenv, os.Getenv)
	}

	// Compute default output path if -o was not given.
	if opts.outputPath == "" {
		opts.outputPath, err = ensureDefaultOutputPath(cfg, opts.outputFormat)
		if err != nil {
			return fmt.Errorf("default output path: %w", err)
		}
	}

	// Read prompt from stdin if available.
	if opts.prompt == "" {
		if stat, sErr := os.Stdin.Stat(); sErr == nil && stat != nil && (stat.Mode()&os.ModeCharDevice) == 0 {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				return fmt.Errorf("read stdin: %w", err)
			}
			opts.prompt = strings.TrimSpace(string(b))
		}
	}

	needTUI := !opts.noTUI && (opts.prompt == "" || token == "" || opts.provider == "")
	if needTUI {
		return runInteractive(start, logger, opts, cfg, token)
	}

	if token == "" {
		return fmt.Errorf("no %s token available; set it in %s, .env, or %s", opts.provider, cfg.Path, envVarFor(opts.provider))
	}
	if strings.TrimSpace(opts.prompt) == "" {
		return errors.New("prompt is required: use -prompt, pipe to stdin, or omit -no-tui")
	}

	resolvedModel := config.ResolveModel(cfg, opts.modelKey, opts.provider)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if opts.timeout > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, opts.timeout)
		defer stop()
	}

	req := buildLibRequest(opts, token, resolvedModel, os.Stderr)
	logger.info("generation.dispatch",
		"provider", req.Provider,
		"model", req.Model,
		"aspect_ratio", req.AspectRatio,
		"output", opts.outputPath,
	)

	res, err := curds.New().Generate(ctx, req)
	if err != nil {
		return err
	}
	if res == nil || len(res.Images) == 0 {
		return errors.New("no images returned")
	}

	paths, err := saveImages(opts, res.Images)
	if err != nil {
		return fmt.Errorf("save: %w", err)
	}

	logger.info("curds.completed",
		"images", len(res.Images),
		"total_bytes", totalBytes(res.Images),
		"duration_ms", time.Since(start).Milliseconds(),
		"paths", strings.Join(paths, ","),
	)

	if shouldShowInline(opts.inline, false) && curds.SupportsInlineImages() {
		showInline(paths, logger)
	}

	if opts.open {
		if err := openInViewer(paths); err != nil {
			logger.error("open.failed", "err", err.Error())
		} else {
			logger.info("open.launched", "viewer", viewerName(), "count", len(paths))
		}
	}
	return nil
}

// shouldShowInline encodes the -inline flag's tristate. tui=true means we're
// running in TUI mode (auto -> on); tui=false means non-TUI (auto -> off).
func shouldShowInline(setting string, tui bool) bool {
	switch strings.ToLower(strings.TrimSpace(setting)) {
	case "on", "true", "yes", "1":
		return true
	case "off", "false", "no", "0":
		return false
	}
	return tui
}

// showInline emits OSC 1337 sequences for each rendered file to stdout.
// Used in non-TUI mode; the TUI handles preview rendering itself.
func showInline(paths []string, logger *logfmtLogger) {
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			logger.error("inline.read_failed", "path", p, "err", err.Error())
			continue
		}
		seq := curds.EncodeInlineImage(data, curds.InlineImageOpts{
			Name:           filepath.Base(p),
			PreserveAspect: true,
		})
		fmt.Print(seq)
		fmt.Println()
	}
}

// runInteractive owns the full TUI flow: clear screen → optional token
// capture → bubbletea main loop with banner / prompt / spinner+logs /
// result / "generate another" loop.
func runInteractive(start time.Time, logger *logfmtLogger, opts *cliOptions, cfg *config.Config, token string) error {
	logger.info("tui.start", "reason", tuiReason(opts.prompt == "", token == "", opts.provider == ""))

	tui.ClearScreen()

	provider := fallback(opts.provider, "openai")

	if token == "" {
		tui.RenderBanner()
		t, err := tui.RunTokenCapture(tui.Defaults{
			Provider:  provider,
			NeedToken: true,
		})
		if err != nil {
			return fmt.Errorf("tui token capture: %w", err)
		}
		if t == nil || t.Cancelled {
			return errors.New("cancelled")
		}
		provider = t.Provider
		token = t.Token
		if t.Save {
			switch provider {
			case "openai":
				cfg.Tokens.OpenAI = token
			case "replicate":
				cfg.Tokens.Replicate = token
			}
			if err := cfg.SaveTokens(); err != nil {
				logger.error("config.save_failed", "err", err.Error())
			} else {
				logger.info("config.token_saved", "provider", provider, "path", cfg.Path)
			}
		}
		tui.ClearScreen()
	}

	opts.provider = provider
	resolvedModel := config.ResolveModel(cfg, opts.modelKey, opts.provider)

	defaults := tui.Defaults{
		Provider:      opts.provider,
		Token:         token,
		AspectRatio:   opts.aspectRatio,
		Quality:       opts.quality,
		NumImages:     opts.numImages,
		OutputFormat:  opts.outputFormat,
		OutputPath:    opts.outputPath,
		InlinePreview: shouldShowInline(opts.inline, true) && curds.SupportsInlineImages(),
	}

	gen := func(ctx context.Context, req tui.GenerateRequest, logsink io.Writer) tui.GenerateResult {
		// Apply per-iteration overrides from the form.
		opts.prompt = req.Prompt
		opts.provider = req.Provider
		opts.aspectRatio = req.AspectRatio
		opts.quality = req.Quality
		opts.numImages = req.NumImages
		opts.outputFormat = req.OutputFormat

		// Refresh output path each iteration so the timestamp is current.
		var err error
		opts.outputPath, err = ensureDefaultOutputPath(cfg, opts.outputFormat)
		if err != nil {
			return tui.GenerateResult{Err: err}
		}

		// Apply timeout from CLI flags.
		callCtx := ctx
		if opts.timeout > 0 {
			var stop context.CancelFunc
			callCtx, stop = context.WithTimeout(ctx, opts.timeout)
			defer stop()
		}

		// Logs go through both the on-screen panel and the CLI logger so the
		// terminal still shows curds.completed when the TUI exits.
		libReq := buildLibRequest(opts, req.Token, resolvedModel, logsink)

		res, err := curds.New().Generate(callCtx, libReq)
		if err != nil {
			return tui.GenerateResult{Err: err}
		}
		if res == nil || len(res.Images) == 0 {
			return tui.GenerateResult{Err: errors.New("no images returned")}
		}
		paths, err := saveImages(opts, res.Images)
		if err != nil {
			return tui.GenerateResult{Err: err}
		}
		fmt.Fprint(logsink, curds.FormatLogLine(
			"info", "curds.completed",
			[]any{
				"images", len(res.Images),
				"total_bytes", totalBytes(res.Images),
				"duration_ms", time.Since(start).Milliseconds(),
				"paths", strings.Join(paths, ","),
			},
			false, // log panel doesn't ANSI-render the inline color
		))
		if opts.open {
			if oerr := openInViewer(paths); oerr != nil {
				fmt.Fprint(logsink, curds.FormatLogLine("error", "open.failed",
					[]any{"err", oerr.Error()}, false))
			} else {
				fmt.Fprint(logsink, curds.FormatLogLine("info", "open.launched",
					[]any{"viewer", viewerName(), "count", len(paths)}, false))
			}
		}
		return tui.GenerateResult{Paths: paths}
	}

	if err := tui.RunInteractive(defaults, gen); err != nil {
		return fmt.Errorf("tui: %w", err)
	}
	return nil
}

func buildLibRequest(opts *cliOptions, token, model string, logger io.Writer) *curds.Request {
	return &curds.Request{
		Provider:          opts.provider,
		Token:             token,
		Model:             model,
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
		Logger:            logger,
		Verbose:           opts.verbose,
	}
}

func totalBytes(images []curds.Image) int {
	t := 0
	for _, img := range images {
		t += len(img.Bytes)
	}
	return t
}

// openInViewer launches the generated images in the OS image viewer.
// macOS uses `open -a Preview`, Linux uses `xdg-open`, Windows uses `start`.
// On unsupported platforms it returns an error rather than silently no-oping.
func openInViewer(paths []string) error {
	if len(paths) == 0 {
		return nil
	}
	switch runtime.GOOS {
	case "darwin":
		args := append([]string{"-a", "Preview"}, paths...)
		return exec.Command("open", args...).Run()
	case "linux":
		// xdg-open opens one file at a time; fan out.
		for _, p := range paths {
			if err := exec.Command("xdg-open", p).Start(); err != nil {
				return err
			}
		}
		return nil
	case "windows":
		for _, p := range paths {
			if err := exec.Command("cmd", "/C", "start", "", p).Start(); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}

func viewerName() string {
	switch runtime.GOOS {
	case "darwin":
		return "Preview"
	case "linux":
		return "xdg-open"
	case "windows":
		return "shell start"
	}
	return runtime.GOOS
}

func parseFlags() (*cliOptions, error) {
	opts := &cliOptions{}

	flag.StringVar(&opts.provider, "provider", "", "Provider: openai, replicate (default: from config or auto-detect)")
	flag.StringVar(&opts.tokenFlag, "token", "", "Provider API token (overrides config/.env/env)")
	flag.StringVar(&opts.modelKey, "model", "", "Model key from config (default: config.default_model)")
	flag.StringVar(&opts.prompt, "prompt", "", "Prompt text (reads stdin if omitted; otherwise launches TUI)")
	flag.StringVar(&opts.outputPath, "output", "", "Output file path (default: <output.directory>/<ms>.<format>)")

	flag.StringVar(&opts.aspectRatio, "aspect-ratio", "", "Aspect ratio override (default: config.defaults.aspect_ratio)")
	flag.StringVar(&opts.size, "size", "", "Explicit pixel size for openai (e.g. 2048x1152)")
	flag.StringVar(&opts.quality, "quality", "", "Quality: low, medium, high, auto")
	flag.IntVar(&opts.numImages, "number-of-images", 0, "Number of images (1-10)")
	flag.StringVar(&opts.outputFormat, "output-format", "", "Output format: webp, png, jpeg")
	flag.IntVar(&opts.outputCompression, "output-compression", -1, "Output compression 0-100 (openai webp/jpeg)")
	flag.StringVar(&opts.background, "background", "", "Background: auto, opaque")
	flag.StringVar(&opts.moderation, "moderation", "", "Moderation: auto, low")
	flag.StringVar(&opts.user, "user", "", "End-user identifier (openai only)")
	flag.StringVar(&opts.replicateBYOKey, "replicate-openai-api-key", "", "BYO OpenAI key for Replicate (replicate provider only)")

	flag.Var(&opts.inputImages, "input-image", fmt.Sprintf("Input reference image(s); repeat or comma-separate, up to %d", curds.MaxInputImages))
	flag.StringVar(&opts.mask, "mask", "", "Mask image file (openai edits only)")

	flag.DurationVar(&opts.pollInterval, "poll-interval", 2*time.Second, "Polling interval for replicate")
	flag.DurationVar(&opts.timeout, "timeout", 10*time.Minute, "Overall timeout (0 disables)")
	flag.BoolVar(&opts.verbose, "verbose", false, "Verbose debug logs to stderr")
	flag.BoolVar(&opts.noTUI, "no-tui", false, "Never enter interactive TUI; fail with an error instead")
	flag.BoolVar(&opts.open, "open", false, "Open generated images in the OS default viewer (macOS: Preview)")
	flag.StringVar(&opts.inline, "inline", "auto", "Show generated images inline in the terminal: auto|on|off (auto = on in TUI, off otherwise)")

	setupUsage()
	flag.Parse()

	opts.provider = strings.ToLower(strings.TrimSpace(opts.provider))
	if opts.outputCompression > 100 {
		return nil, fmt.Errorf("output-compression must be 0-100, got %d", opts.outputCompression)
	}
	return opts, nil
}

func setupUsage() {
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		fmt.Fprint(out, helpText())
	}
}

// helpText returns the full help. Structured for both LLMs (predictable
// section headers, types and allowed values inline, complete examples) and
// humans (man-page conventions, no Markdown noise on stdout).
func helpText() string {
	return `NAME
  curds — generate images from text prompts via OpenAI or Replicate

SYNOPSIS
  curds [flags]
  curds -prompt PROMPT [flags]
  echo PROMPT | curds [flags]

DESCRIPTION
  Generates images using gpt-image-2 (default). Saves to
  ~/Desktop/curds/<unix_milli>.<format> unless -o is given. Auto-creates
  ~/.config/curds/config.toml on first run. Drops into an interactive TUI
  when prompt or token is missing (suppress with -no-tui).

PROVIDERS
  openai     OpenAI Image API direct
             Endpoints: POST /v1/images/generations
                        POST /v1/images/edits   (when -i or -mask is set)
             Default model: gpt-image-2
  replicate  Replicate hosted
             Endpoint:  POST /v1/models/<owner>/<name>/predictions
             Default model: openai/gpt-image-2

  Provider auto-detect order: -provider flag > config.provider >
  whichever token is available (openai preferred when both are present).

TOKEN RESOLUTION (first non-empty wins)
  1. -token flag
  2. ~/.config/curds/config.toml [tokens] section
  3. .env file in cwd
  4. environment: OPENAI_API_KEY (openai), REPLICATE_API_TOKEN (replicate)

EXIT CODES
  0  success
  1  upstream error, generation failed, or user cancelled
  2  invalid flag combination or missing required input

FLAGS

  Core
    -prompt   STRING            prompt text. If omitted, reads stdin when
                                piped, otherwise opens the TUI.
    -output   PATH              output file path
                                default: ~/Desktop/curds/<unix_milli>.<format>
                                extension drives -output-format unless set
    -no-tui                     never enter interactive TUI; fail instead
    -open                       open generated images in OS image viewer
                                (macOS: Preview, linux: xdg-open, win: start)
    -inline   {auto|on|off}     show images inline in the terminal
                                (auto = on in TUI mode if supported, off
                                otherwise; supported terminals: iTerm2,
                                WezTerm, VS Code, Konsole, Tabby, Ghostty)
    -verbose                    include debug-level logs on stderr
    -timeout  DURATION          overall timeout (default 10m, 0 disables)

  Provider & auth
    -provider {openai|replicate}      backend (default: auto-detect)
    -token    STRING                  API token (overrides config/.env/env)
    -model    KEY                     model key from config
                                      (default: config.default_model)
    -user     STRING                  end-user identifier (openai only)
    -replicate-openai-api-key STRING  BYO OpenAI key passed through
                                      Replicate (replicate only)

  Image parameters
    -aspect-ratio       RATIO              see ASPECT RATIOS (default: 16:9)
    -size               WxH                explicit pixel size (openai)
                                           rounded to gpt-image-2 constraints
    -quality            {low|medium|high|auto}     default: auto
    -number-of-images   N                  1-10 (default: 1)
    -output-format      {webp|png|jpeg}    default: webp
    -output-compression 0-100              openai webp/jpeg (default: 90)
    -background         {auto|opaque}      default: auto
                                           (transparent unsupported by
                                           gpt-image-2)
    -moderation         {auto|low}         default: auto

  Reference images / edits
    -input-image PATH           reference image(s); repeat or comma-separate,
                                up to 16 total. Accepts file paths,
                                http(s) URLs, or data: URLs.
                                OpenAI switches to /v1/images/edits.
    -mask        PATH           mask file (openai edits only,
                                PNG with alpha channel)

  Replicate-specific
    -poll-interval DURATION     status poll cadence (default: 2s)

ASPECT RATIOS
  Replicate gpt-image-2 accepts only:  1:1, 3:2, 2:3

  OpenAI gpt-image-2 constraints:
    - both edges multiples of 16
    - max edge <= 3840 px
    - long edge / short edge <= 3:1
    - total pixels in [655,360 .. 8,294,400]

  Named ratios (mapped to multiples-of-16 sizes):
    1:1        1024x1024
    3:2        1536x1024
    2:3        1024x1536
    4:3        1536x1152
    3:4        1152x1536
    16:9       2048x1152    ~1080p+ landscape (default)
    9:16       1152x2048    ~1080p+ portrait
    21:9       2688x1152
    9:21       1152x2688
    2:1        2048x1024
    1:2        1024x2048
    16:9-4k    3840x2160
    9:16-4k    2160x3840

  Custom -size values are rounded automatically:
    -size 1920x1080  ->  1920x1088   (1080 isn't on the 16-pixel grid)
    -size 5000x1000  ->  3024x1008   (clamped to max edge then 3:1 ratio)

LOGGING
  Every stage writes a logfmt event to stderr:
    ts=...  level={info|error|debug}  event=NAME  key=value ...
  TTY:    output is colorized
  Pipes:  plain logfmt, safe for log collectors
  -v:     include debug-level events (request bodies, polling cadence)

CONFIG FILE
  Path:    ~/.config/curds/config.toml
  Override: $CURDS_CONFIG
  Schema:  see "default_model", [output], [tokens], [defaults], [models.<key>]
           Auto-written on first run; tokens added by the TUI when you say so.

EXAMPLES
  # Simplest — uses 16:9 default + auto-detected provider
  curds -prompt "a watercolor fox in a meadow"

  # Pipe a long prompt from a file
  cat prompt.txt | curds -aspect-ratio 9:16 -quality high

  # Force OpenAI, ~1080p landscape, custom output
  curds -provider openai -aspect-ratio 16:9 \
        -prompt "neon cyberpunk city skyline" -output /tmp/city.webp

  # Compose from reference images (OpenAI /v1/images/edits)
  curds -input-image ref1.png,ref2.png \
        -prompt "combine into a gift basket on white"

  # Inpainting via mask (OpenAI only — mask is PNG with alpha)
  curds -input-image lounge.png -mask mask.png \
        -prompt "add a flamingo in the pool"

  # Generate 4 variants in one call
  curds -prompt "watercolor fox" -number-of-images 4

  # Scripted use — never enter TUI
  curds -no-tui -prompt "$PROMPT" -output "$OUT"

  # Generate and open in Preview (macOS)
  curds -open -prompt "a watercolor fox in a meadow"

FILES
  ~/.config/curds/config.toml    config (auto-created)
  ~/Desktop/curds/               default output directory (auto-created)
  .env                           optional; read from cwd

ENVIRONMENT
  CURDS_CONFIG                   override config file path
  OPENAI_API_KEY                 fallback OpenAI token
  REPLICATE_API_TOKEN            fallback Replicate token
  XDG_CONFIG_HOME                respected for default config path
`
}

func applyConfigDefaults(opts *cliOptions, cfg *config.Config) {
	if opts.provider == "" && cfg.Provider != "" {
		opts.provider = cfg.Provider
	}
	if opts.modelKey == "" {
		opts.modelKey = cfg.DefaultModel
	}
	if opts.aspectRatio == "" {
		opts.aspectRatio = cfg.Defaults.AspectRatio
	}
	if opts.quality == "" {
		opts.quality = cfg.Defaults.Quality
	}
	if opts.numImages == 0 {
		opts.numImages = cfg.Defaults.NumberOfImages
	}
	if opts.outputFormat == "" {
		opts.outputFormat = cfg.Output.Format
	}
	if opts.outputCompression == -1 {
		opts.outputCompression = cfg.Output.Compression
	}
	if opts.background == "" {
		opts.background = cfg.Defaults.Background
	}
	if opts.moderation == "" {
		opts.moderation = cfg.Defaults.Moderation
	}

	// If -o has an extension and -output-format wasn't explicitly set, derive
	// the format from the extension so the bytes match the filename.
	if opts.outputPath != "" && !flagWasSet("output-format") {
		if ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(opts.outputPath), ".")); ext != "" {
			switch ext {
			case "webp", "png":
				opts.outputFormat = ext
			case "jpeg", "jpg":
				opts.outputFormat = "jpeg"
			}
		}
	}
}

func ensureDefaultOutputPath(cfg *config.Config, format string) (string, error) {
	dir := config.ExpandTilde(cfg.Output.Directory)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	if format == "" {
		format = "webp"
	}
	name := fmt.Sprintf("%d.%s", time.Now().UnixMilli(), format)
	return filepath.Join(dir, name), nil
}

func saveImages(opts *cliOptions, images []curds.Image) ([]string, error) {
	origExt := filepath.Ext(opts.outputPath)
	stem := strings.TrimSuffix(opts.outputPath, origExt)
	ext := origExt
	if ext == "" {
		ext = "." + opts.outputFormat
	}

	paths := make([]string, 0, len(images))
	for i, img := range images {
		var path string
		switch {
		case len(images) > 1:
			path = fmt.Sprintf("%s-%d%s", stem, i+1, ext)
		case origExt == "":
			path = stem + ext
		default:
			path = opts.outputPath
		}
		if dir := filepath.Dir(path); dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, err
			}
		}
		if err := os.WriteFile(path, img.Bytes, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		if opts.verbose && img.RevisedPrompt != "" {
			fmt.Fprint(os.Stderr, curds.FormatLogLine(
				"info", "openai.revised_prompt",
				[]any{"index", i, "prompt", img.RevisedPrompt},
				curds.IsTerminalWriter(os.Stderr),
			))
		}
		fmt.Println(path)
		paths = append(paths, path)
	}
	return paths, nil
}

func flagWasSet(name string) bool {
	seen := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			seen = true
		}
	})
	return seen
}

func envVarFor(provider string) string {
	switch provider {
	case "openai":
		return "OPENAI_API_KEY"
	case "replicate":
		return "REPLICATE_API_TOKEN"
	}
	return "(unknown)"
}

func fallback(s, def string) string {
	if s != "" {
		return s
	}
	return def
}

func tuiReason(noPrompt, noToken, noProvider bool) string {
	parts := []string{}
	if noProvider {
		parts = append(parts, "no_provider")
	}
	if noToken {
		parts = append(parts, "no_token")
	}
	if noPrompt {
		parts = append(parts, "no_prompt")
	}
	return strings.Join(parts, ",")
}

// ============================================================================
// CLI logger — uses the shared formatter so library and CLI events look
// identical and TTY/color detection is in one place.
// ============================================================================

type logfmtLogger struct {
	w     io.Writer
	color bool
}

func newLogger(w io.Writer) *logfmtLogger {
	return &logfmtLogger{w: w, color: curds.IsTerminalWriter(w)}
}

func (l *logfmtLogger) info(event string, kv ...any)  { l.write("info", event, kv...) }
func (l *logfmtLogger) error(event string, kv ...any) { l.write("error", event, kv...) }

func (l *logfmtLogger) write(level, event string, kv ...any) {
	if l == nil || l.w == nil {
		return
	}
	fmt.Fprint(l.w, curds.FormatLogLine(level, event, kv, l.color))
}
