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
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gersham/curds"
	"github.com/gersham/curds/config"
	"github.com/gersham/curds/tui"
)

// version is the curds release version, reported by the curds.start log event.
const version = "0.2.0"

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
	lastFrameImage    string
	referenceImages   imageList
	referenceVideos   imageList
	referenceAudios   imageList
	imageResolution   string
	videoDuration     int
	videoResolution   string
	noAudio           bool
	stripAudio        bool
	seed              int
	scale             float64
	faceEnhance       bool
	pollInterval      time.Duration
	timeout           time.Duration
	verbose           bool
	noTUI             bool
	open              bool
	inline            string // "auto" | "on" | "off"
	showVersion       bool
}

func main() {
	logger := newLogger(os.Stderr)
	start := time.Now()

	if err := realMain(logger, start); err != nil {
		logger.error("curds.failed", "err", err.Error(), "duration_ms", time.Since(start).Milliseconds())
		fmt.Fprintln(os.Stderr, "error:", err)
		// Usage problems exit 2, per the EXIT CODES section of the help.
		var ue *usageError
		if errors.As(err, &ue) {
			os.Exit(2)
		}
		os.Exit(1)
	}
}

func realMain(logger *logfmtLogger, start time.Time) error {
	logger.info("curds.start", "version", version)

	opts, err := parseFlags()
	if err != nil {
		// An explicit -help/-h is not a failure: print the help, exit 0.
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(os.Stderr, helpText())
			return nil
		}
		// Anything else is a usage error. Do not dump the full help over it —
		// a 40-line usage wall buries the one line that says what is wrong.
		return err
	}

	if opts.showVersion {
		fmt.Println("curds " + version)
		return nil
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

	// For mp4 output with no explicit -model, the configured video model
	// (Replicate-hosted Seedance 2.0 by default) wins. Fall back to xAI's
	// native grok-imagine-video only when that model needs a replicate token we
	// don't have and an xai token is available.
	if !flagWasSet("model") && opts.outputFormat == "mp4" && !flagWasSet("provider") {
		needsReplicate := true
		if m, ok := cfg.Models[opts.modelKey]; ok && m.ReplicateName == "" {
			needsReplicate = false
		}
		if needsReplicate && config.ResolveToken("replicate", cfg, dotenv, os.Getenv) == "" &&
			config.ResolveToken("xai", cfg, dotenv, os.Getenv) != "" {
			logger.info("video.model_fallback", "from", opts.modelKey, "to", curds.DefaultXaiVideoModel, "reason", "no replicate token")
			opts.modelKey = curds.DefaultXaiVideoModel
		}
	}

	// Resolve provider.
	if opts.provider == "" {
		opts.provider = config.DetectProvider(cfg, dotenv, os.Getenv)
	}
	// If the user picked a model whose config entry binds to a single provider
	// (e.g. seedance-2/remove-bg → replicate, grok-imagine-video → xai), force
	// that provider. Without this, auto-detect could pick the wrong one and
	// ResolveModel would pass the raw key through as if it belonged to it.
	if !flagWasSet("provider") && opts.modelKey != "" {
		if m, ok := cfg.Models[opts.modelKey]; ok {
			switch {
			case m.OpenAIName == "" && m.ReplicateName == "" && m.XaiName != "":
				opts.provider = "xai"
			case m.OpenAIName == "" && m.ReplicateName != "":
				opts.provider = "replicate"
			}
		}
	}

	// Resolve token if user didn't pass -token. CLI flag overrides everything.
	token := opts.tokenFlag
	if token == "" && opts.provider != "" {
		token = config.ResolveToken(opts.provider, cfg, dotenv, os.Getenv)
	}

	// Resolve the model now (before the TUI gate) so model-aware defaults
	// like png-output for segmentation are applied to opts.outputFormat
	// before we compute the default output path.
	resolvedModel := config.ResolveModel(cfg, opts.modelKey, opts.provider)
	applyModelOutputDefaults(opts, cfg, resolvedModel, logger)

	// Compute default output path if -output was not given.
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

	// Segmentation (bria/remove-background) and upscale (real-esrgan) models
	// take an input image instead of a prompt — don't drop into the
	// prompt-asking TUI for them.
	needsPrompt := !curds.IsSegmentationModel(resolvedModel) && !curds.IsUpscaleModel(resolvedModel)
	needTUI := !opts.noTUI && (token == "" || opts.provider == "" || (needsPrompt && opts.prompt == ""))
	if needTUI {
		return runInteractive(start, logger, opts, cfg, token)
	}

	if token == "" {
		return fmt.Errorf("no %s token available; set it in %s, .env, or %s", opts.provider, cfg.Path, envVarFor(opts.provider))
	}
	if needsPrompt && strings.TrimSpace(opts.prompt) == "" {
		return errors.New("prompt is required: use -prompt, pipe to stdin, or omit -no-tui")
	}
	if !needsPrompt && len(opts.inputImages) == 0 {
		return errors.New("this model requires -input-image PATH")
	}

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
	if res == nil || (len(res.Images) == 0 && len(res.Videos) == 0) {
		return errors.New("no assets returned")
	}

	paths, err := saveResult(opts, res)
	if err != nil {
		return fmt.Errorf("save: %w", err)
	}

	maybeStripAudio(opts, len(res.Videos), paths, os.Stderr)

	logger.info("curds.completed",
		"images", len(res.Images),
		"videos", len(res.Videos),
		"total_bytes", totalResultBytes(res),
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
			case "xai":
				cfg.Tokens.Xai = token
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
	applyModelOutputDefaults(opts, cfg, resolvedModel, logger)

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

	settings := tui.SettingsValues{
		Provider:          cfg.Provider,
		OutputDirectory:   cfg.Output.Directory,
		OutputFormat:      cfg.Output.Format,
		OutputCompression: cfg.Output.Compression,
		OpenAIToken:       cfg.Tokens.OpenAI,
		ReplicateToken:    cfg.Tokens.Replicate,
		Quality:           cfg.Defaults.Quality,
		AspectRatio:       cfg.Defaults.AspectRatio,
		Background:        cfg.Defaults.Background,
		Moderation:        cfg.Defaults.Moderation,
		NumberOfImages:    cfg.Defaults.NumberOfImages,
	}

	saveSettings := func(s tui.SettingsValues) error {
		cfg.Provider = s.Provider
		cfg.Output.Directory = s.OutputDirectory
		cfg.Output.Format = s.OutputFormat
		cfg.Output.Compression = s.OutputCompression
		cfg.Tokens.OpenAI = s.OpenAIToken
		cfg.Tokens.Replicate = s.ReplicateToken
		cfg.Defaults.Quality = s.Quality
		cfg.Defaults.AspectRatio = s.AspectRatio
		cfg.Defaults.Background = s.Background
		cfg.Defaults.Moderation = s.Moderation
		cfg.Defaults.NumberOfImages = s.NumberOfImages

		// Mirror the same fields onto the live opts so the *next* generation
		// (without restarting curds) picks up the new defaults.
		opts.provider = s.Provider
		opts.aspectRatio = s.AspectRatio
		opts.quality = s.Quality
		opts.numImages = s.NumberOfImages
		opts.outputFormat = s.OutputFormat
		opts.outputCompression = s.OutputCompression
		opts.background = s.Background
		opts.moderation = s.Moderation

		// Token used by the generate callback comes through req.Token, which
		// the TUI rebuilds from settings.OpenAIToken / .ReplicateToken on
		// each call — so updating cfg + opts is enough.
		if err := cfg.Save(); err != nil {
			logger.error("config.save_failed", "err", err.Error())
			return err
		}
		logger.info("config.saved", "path", cfg.Path)
		return nil
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
		if res == nil || (len(res.Images) == 0 && len(res.Videos) == 0) {
			return tui.GenerateResult{Err: errors.New("no assets returned")}
		}
		paths, err := saveResult(opts, res)
		if err != nil {
			return tui.GenerateResult{Err: err}
		}
		maybeStripAudio(opts, len(res.Videos), paths, logsink)
		fmt.Fprint(logsink, curds.FormatLogLine(
			"info", "curds.completed",
			[]any{
				"images", len(res.Images),
				"videos", len(res.Videos),
				"total_bytes", totalResultBytes(res),
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

	if err := tui.RunInteractive(defaults, gen, settings, saveSettings); err != nil {
		return fmt.Errorf("tui: %w", err)
	}
	return nil
}

func buildLibRequest(opts *cliOptions, token, model string, logger io.Writer) *curds.Request {
	audio := !opts.noAudio
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
		LastFrameImage:    opts.lastFrameImage,
		ReferenceImages:   []string(opts.referenceImages),
		ReferenceVideos:   []string(opts.referenceVideos),
		ReferenceAudios:   []string(opts.referenceAudios),
		VideoDuration:     opts.videoDuration,
		ImageResolution:   opts.imageResolution,
		VideoResolution:   opts.videoResolution,
		GenerateAudio:     &audio,
		Seed:              opts.seed,
		Scale:             opts.scale,
		FaceEnhance:       opts.faceEnhance,
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

func totalVideoBytes(videos []curds.Video) int {
	t := 0
	for _, video := range videos {
		t += len(video.Bytes)
	}
	return t
}

func totalResultBytes(res *curds.Result) int {
	if res == nil {
		return 0
	}
	return totalBytes(res.Images) + totalVideoBytes(res.Videos)
}

// openInViewer launches the generated assets in the OS viewer.
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

	// ContinueOnError + a discarded writer hands us the error instead of the
	// stdlib printing a terse one-liner, dumping the whole help, and exiting.
	// We render both the message and the help ourselves.
	flag.CommandLine.Init("curds", flag.ContinueOnError)
	flag.CommandLine.SetOutput(io.Discard)

	flag.StringVar(&opts.provider, "provider", "", "Provider: openai, replicate, xai (default: from config or auto-detect)")
	flag.StringVar(&opts.tokenFlag, "token", "", "Provider API token (overrides config/.env/env)")
	flag.StringVar(&opts.modelKey, "model", "", "Model key from config (default: config.default_model, or config.default_video_model for mp4 output)")
	flag.StringVar(&opts.prompt, "prompt", "", "Prompt text (reads stdin if omitted; otherwise launches TUI)")
	flag.StringVar(&opts.outputPath, "output", "", "Output file path (default: <output.directory>/<ms>.<format>)")

	flag.StringVar(&opts.aspectRatio, "aspect-ratio", "", "Aspect ratio override (default: config.defaults.aspect_ratio)")
	flag.StringVar(&opts.size, "size", "", "Explicit pixel size for openai (e.g. 2048x1152)")
	flag.StringVar(&opts.quality, "quality", "", "Quality: low, medium, high, auto")
	flag.IntVar(&opts.numImages, "number-of-images", 0, "Number of images (1-10)")
	flag.StringVar(&opts.outputFormat, "output-format", "", "Output format: webp, png, jpeg, mp4")
	flag.IntVar(&opts.outputCompression, "output-compression", -1, "Output compression 0-100 (openai webp/jpeg)")
	flag.StringVar(&opts.background, "background", "", "Background: auto, opaque")
	flag.StringVar(&opts.moderation, "moderation", "", "Moderation: auto, low")
	flag.StringVar(&opts.user, "user", "", "End-user identifier (openai only)")
	flag.StringVar(&opts.replicateBYOKey, "replicate-openai-api-key", "", "BYO OpenAI key for Replicate (replicate provider only)")

	flag.Var(&opts.inputImages, "input-image", fmt.Sprintf("Input reference image(s); repeat or comma-separate, up to %d", curds.MaxInputImages))
	flag.StringVar(&opts.mask, "mask", "", "Mask image file (openai edits only)")
	flag.StringVar(&opts.lastFrameImage, "last-frame-image", "", "Seedance last-frame image (requires one -input-image first frame)")
	flag.Var(&opts.referenceImages, "reference-image", "Reference image(s); repeat or comma-separate (MiniMax H3 and Seedance up to 9; xai grok-imagine-video)")
	flag.Var(&opts.referenceVideos, "reference-video", "Seedance reference video(s); repeat or comma-separate, up to 3")
	flag.Var(&opts.referenceAudios, "reference-audio", "Seedance reference audio(s); repeat or comma-separate, up to 3")
	flag.IntVar(&opts.videoDuration, "video-duration", 0, "Video duration in seconds: Grok/xai 1-15; Seedance -1 or 4-15 (default: 5)")
	flag.StringVar(&opts.imageResolution, "image-resolution", "", "Image resolution for models that size by target: flux-2-pro 0.5mp/1mp/2mp/4mp; nano-banana-2 1k/2k/4k")
	flag.StringVar(&opts.videoResolution, "video-resolution", "", "Video resolution: Kling 720p/1080p/4k (default: 1080p); MiniMax H3 768p/2k (default: 768p); Grok 480p/720p; xai 480p/720p; Seedance 480p/720p/1080p (default: 720p)")
	flag.BoolVar(&opts.noAudio, "no-audio", false, "Disable Seedance synchronized audio generation")
	flag.BoolVar(&opts.stripAudio, "strip-audio", true, "Strip the audio track from generated videos via ffmpeg if installed (default: true)")
	flag.IntVar(&opts.seed, "seed", 0, "Random seed for supported Replicate models (0 = random)")
	flag.Float64Var(&opts.scale, "scale", 0, "Upscale factor: -model upscale 1-10 (default: 4); -model upscale-pro 2, 4, or 6")
	flag.BoolVar(&opts.faceEnhance, "face-enhance", false, "Run GFPGAN face enhancement (upscale models only)")

	flag.DurationVar(&opts.pollInterval, "poll-interval", 2*time.Second, "Polling interval for replicate")
	flag.DurationVar(&opts.timeout, "timeout", 10*time.Minute, "Overall timeout (0 disables)")
	flag.BoolVar(&opts.verbose, "verbose", false, "Verbose debug logs to stderr")
	flag.BoolVar(&opts.noTUI, "no-tui", false, "Never enter interactive TUI; fail with an error instead")
	flag.BoolVar(&opts.open, "open", false, "Open generated assets in the OS default viewer (macOS: Preview)")
	flag.StringVar(&opts.inline, "inline", "auto", "Show generated images inline in the terminal: auto|on|off (auto = on in TUI, off otherwise)")
	flag.BoolVar(&opts.showVersion, "version", false, "Print the curds version and exit")

	setupUsage()
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil, err
		}
		return nil, &usageError{err: explainFlagError(err)}
	}

	opts.provider = strings.ToLower(strings.TrimSpace(opts.provider))
	if opts.outputCompression > 100 {
		return nil, &usageError{err: fmt.Errorf("output-compression must be 0-100, got %d", opts.outputCompression)}
	}
	return opts, nil
}

// usageError marks a command-line usage problem so main() can exit 2, which is
// what the EXIT CODES section of the help promises.
type usageError struct{ err error }

func (e *usageError) Error() string { return e.err.Error() }
func (e *usageError) Unwrap() error { return e.err }

// undefinedFlagPrefix is the stdlib flag package's wording for an unknown flag.
// Matching on it is the only way to recover the name the user actually typed —
// the package does not expose it. If a future Go release reworks the string we
// fall through to returning the error unchanged, which is still correct.
const undefinedFlagPrefix = "flag provided but not defined: -"

// explainFlagError turns a stdlib flag error into one actionable line. For an
// unknown flag it names the real flags that look closest, because the failure
// we keep hitting is a plausible-looking short form (-o, -n, -v) that curds
// deliberately does not define: one canonical long name per flag.
func explainFlagError(err error) error {
	msg := err.Error()
	if !strings.HasPrefix(msg, undefinedFlagPrefix) {
		return err
	}
	typed := strings.TrimPrefix(msg, undefinedFlagPrefix)
	if near := suggestFlags(typed, definedFlagNames()); len(near) > 0 {
		return fmt.Errorf("unknown flag -%s; did you mean %s? (curds -help lists every flag)",
			typed, joinOr(near))
	}
	return fmt.Errorf("unknown flag -%s; curds -help lists every flag", typed)
}

// definedFlagNames returns every registered flag name.
func definedFlagNames() []string {
	var names []string
	flag.VisitAll(func(f *flag.Flag) { names = append(names, f.Name) })
	return names
}

// maxSuggestions caps the candidate list so the message stays one line.
const maxSuggestions = 3

// suggestFlags ranks defined flag names by how plausibly they are what `typed`
// meant. A name qualifies if `typed` is a prefix of it (covers -o, -n, -out) or
// if it is within a small edit distance (covers typos like -promt). Ranking is
// edit distance, then length, then alphabetical, so the ordering is stable.
func suggestFlags(typed string, defined []string) []string {
	typed = strings.ToLower(typed)
	if typed == "" {
		return nil
	}
	// Scale the typo tolerance to the input: one edit on a short flag turns it
	// into a different real flag, so only allow that on longer names.
	tolerance := 1
	if len(typed) >= 5 {
		tolerance = 2
	}

	type cand struct {
		name string
		dist int
	}
	var cands []cand
	for _, name := range defined {
		lower := strings.ToLower(name)
		d := levenshtein(typed, lower)
		// A shared prefix catches near-misses that edit distance does not:
		// "version" and "verbose" are 4 edits apart but share "ver".
		if strings.HasPrefix(lower, typed) || d <= tolerance || commonPrefixLen(lower, typed) >= minSharedPrefix {
			cands = append(cands, cand{name: name, dist: d})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].dist != cands[j].dist {
			return cands[i].dist < cands[j].dist
		}
		if len(cands[i].name) != len(cands[j].name) {
			return len(cands[i].name) < len(cands[j].name)
		}
		return cands[i].name < cands[j].name
	})
	if len(cands) > maxSuggestions {
		cands = cands[:maxSuggestions]
	}
	out := make([]string, 0, len(cands))
	for _, c := range cands {
		out = append(out, "-"+c.name)
	}
	return out
}

// minSharedPrefix is how many leading characters two names must share before
// one is offered as a suggestion for the other.
const minSharedPrefix = 3

// commonPrefixLen returns the number of leading runes a and b share.
func commonPrefixLen(a, b string) int {
	ar, br := []rune(a), []rune(b)
	n := 0
	for n < len(ar) && n < len(br) && ar[n] == br[n] {
		n++
	}
	return n
}

// joinOr renders a candidate list as prose: "-a", "-a or -b", "-a, -b or -c".
func joinOr(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	}
	return strings.Join(items[:len(items)-1], ", ") + " or " + items[len(items)-1]
}

// levenshtein returns the edit distance between a and b.
func levenshtein(a, b string) int {
	if a == b {
		return 0
	}
	ar, br := []rune(a), []rune(b)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		for j := 1; j <= len(br); j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			curr[j] = min(prev[j]+1, min(curr[j-1]+1, prev[j-1]+cost))
		}
		prev, curr = curr, prev
	}
	return prev[len(br)]
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
  curds — generate images and videos from text prompts via OpenAI, Replicate, or xAI

SYNOPSIS
  curds [flags]
  curds -prompt PROMPT [flags]
  echo PROMPT | curds [flags]

DESCRIPTION
  Generates images using gpt-image-2 (default; FLUX.2 [pro] and Nano Banana 2
  also available on Replicate), videos with Seedance 2.0 on Replicate (default
  for mp4; Kling 3.0, MiniMax H3, Grok Imagine Video also selectable),
  removes backgrounds
  with bria/remove-background (-model remove-bg), or upscales images with
  nightmareai/real-esrgan (-model upscale) on Replicate. Saves to
  ~/Desktop/curds/<unix_milli>.<format> unless -output is given. Auto-creates
  ~/.config/curds/config.toml on first run. Drops into an interactive TUI
  when prompt or token is missing (suppress with -no-tui).

PROVIDERS
  openai     OpenAI Image API direct   [recommended for OpenAI models]
             Endpoints: POST /v1/images/generations
                        POST /v1/images/edits   (when -input-image or -mask is set)
             Default model: gpt-image-2
             Why prefer this: lower latency, lower cost, full parameter
             surface (any valid -size, -output-compression, -user, etc.),
             and immediate b64_json responses (no polling).

  replicate  Replicate hosted
             Endpoint:  POST /v1/models/<owner>/<name>/predictions
             Default image model: openai/gpt-image-2
             Default video model: bytedance/seedance-2.0
             Also available: -model flux-2-pro, nano-banana-2,
                             kling-v3, minimax-h3,
                             grok-imagine-video-1.5, upscale-pro
             Use when: you don't have direct OpenAI access yet, or you
             want to run a non-OpenAI image or video model hosted on Replicate.
             Tradeoffs: extra hop adds latency, the gpt-image-2 wrapper
             restricts -aspect-ratio to 1:1, 3:2, 2:3 only, and there's
             no -size / -output-compression passthrough.

  xai        xAI native video API
             Endpoints: POST /v1/videos/generations
                        GET  /v1/videos/{request_id}   (async polling)
             Default model: grok-imagine-video
             Why prefer this: ~half the Replicate cost at 720p, plus
             text-to-video (image optional), reference images, 480p/720p, and
             durations up to 15s. Video-only — no image generation.
             Used automatically for mp4 when no replicate token is set.

  Provider auto-detect (when -provider is omitted):
    1. config.provider in ~/.config/curds/config.toml
    2. for mp4 output with no -model, config.default_video_model
       (seedance-2 → replicate); xai is used instead when no replicate
       token is available but an xai token is.
    3. token availability — OpenAI is preferred for images when present.

TOKEN RESOLUTION (first non-empty wins)
  1. -token flag
  2. ~/.config/curds/config.toml [tokens] section
  3. .env file in cwd
  4. environment: OPENAI_API_KEY (openai), REPLICATE_API_TOKEN (replicate),
     XAI_API_KEY (xai)

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
    -version                    print the curds version and exit
    -no-tui                     never enter interactive TUI; fail instead
    -open                       open generated assets in OS viewer
                                (macOS: Preview, linux: xdg-open, win: start)
    -inline   {auto|on|off}     show images inline in the terminal
                                (auto = on in TUI mode if supported, off
                                otherwise; supported terminals: iTerm2,
                                WezTerm, VS Code, Konsole, Tabby, Ghostty)
    -verbose                    include debug-level logs on stderr
    -timeout  DURATION          overall timeout (default 10m, 0 disables)

  Provider & auth
    -provider {openai|replicate|xai}  backend (default: auto-detect)
    -token    STRING                  API token (overrides config/.env/env)
    -model    KEY                     model key from config
                                      (default: config.default_model, or
                                      config.default_video_model for mp4)
    -user     STRING                  end-user identifier (openai only)
    -replicate-openai-api-key STRING  BYO OpenAI key passed through
                                      Replicate (replicate only)

  Image parameters
    -aspect-ratio       RATIO              see ASPECT RATIOS (default: 1:1)
    -size               WxH                explicit pixel size (openai)
                                           rounded to gpt-image-2 constraints
    -quality            {low|medium|high|auto}     default: auto
    -number-of-images   N                  1-10 (default: 1)
    -image-resolution   VALUE              flux-2-pro: 0.5mp, 1mp, 2mp, 4mp,
                                           match_input_image (default: 1mp);
                                           nano-banana-2: 1k, 2k, 4k
                                           (default: 1k). Ignored by other
                                           models.
    -output-format      {webp|png|jpeg|mp4} default: webp, or mp4 for video
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

  Segmentation / background removal
    -model remove-bg            run bria/remove-background on Replicate.
                                Requires exactly one -input-image (file path,
                                http(s) URL, or data URL). No -prompt.
                                Output is a transparent PNG matching the
                                input dimensions. Output format is forced
                                to png and -aspect-ratio / -size are
                                ignored.

  Upscaling / super-resolution
    -model upscale              run nightmareai/real-esrgan on Replicate.
                                Requires exactly one -input-image (file path,
                                http(s) URL, or data URL). No -prompt.
                                Output is an upscaled PNG. Output format is
                                forced to png and -aspect-ratio / -size are
                                ignored.
    -scale N                    upscale factor, 1-10 (default: 4)
    -face-enhance               run GFPGAN face enhancement
    -model upscale-pro          run topazlabs/image-upscale instead — a
                                modern alternative to Real-ESRGAN. -scale
                                accepts 2, 4, or 6 (omit for enhance-only);
                                -face-enhance requires a -scale.

  Video (xai / Replicate)
    -poll-interval DURATION     status poll cadence (default: 2s)
    -video-duration N           Seedance: -1 or 4-15 seconds; Kling: 3-15;
                                MiniMax H3: 4-15; xai/Grok: 1-15
                                (default: 5)
    -video-resolution VALUE      Kling: 720p, 1080p, 4k (default: 1080p);
                                MiniMax H3: 768p, 2k (default: 768p);
                                Grok/xai: 480p, 720p; Seedance also 1080p
                                (default: 720p)
    -no-audio                    disable Seedance / Kling synchronized audio
                                (MiniMax H3 and xai/Grok always emit audio)
    -strip-audio                 remove the audio track from generated
                                videos via ffmpeg (default: true). xai/Grok
                                always emit audio and the x.ai API has no
                                mute option, so this is the way to get silent
                                clips. No-op without ffmpeg on PATH; pass
                                -strip-audio=false to keep audio.
    -seed N                      random seed for supported Replicate models
    -last-frame-image PATH       Seedance / Kling / MiniMax H3 last frame;
                                requires -input-image
    -input-image PATH            Seedance first frame or reference images;
                                Kling / MiniMax H3 first frame (0 or 1);
                                xai/Grok image-to-video source (1)
    -reference-image PATH        MiniMax H3 / Seedance up to 9; xai
                                grok-imagine-video refs
    -reference-video PATH        MiniMax H3 / Seedance reference video(s), up to 3
    -reference-audio PATH        MiniMax H3 / Seedance reference audio(s), up to 3

ASPECT RATIOS
  Replicate gpt-image-2 accepts only:  1:1, 3:2, 2:3
  Grok Imagine Video 1.5 accepts:       auto, 16:9, 4:3, 1:1,
                                       9:16, 3:4, 3:2, 2:3
  xai grok-imagine-video accepts:       auto, 1:1, 16:9, 9:16,
                                       4:3, 3:4, 3:2, 2:3
  Seedance 2.0 accepts:                 16:9, 4:3, 1:1, 3:4,
                                       9:16, 21:9, 9:21, adaptive
  MiniMax H3 accepts:                   21:9, 16:9, 4:3, 1:1, 3:4,
                                       9:16, adaptive
  Kling 3.0 accepts:                    16:9, 9:16, 1:1
  FLUX.2 [pro] accepts:                 match_input_image, 1:1, 16:9,
                                       3:2, 2:3, 4:5, 5:4, 9:16, 3:4, 4:3
                                       (or -size WxH, 256-2048 per edge)
  Nano Banana 2 accepts:                match_input_image, 1:1, 1:4, 1:8,
                                       2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4,
                                       8:1, 9:16, 16:9, 21:9

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
    16:9       2048x1152    ~1080p+ landscape
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
  -verbose: include debug-level events (request bodies, polling cadence)

CONFIG FILE
  Path:    ~/.config/curds/config.toml
  Override: $CURDS_CONFIG
  Schema:  see "default_model", [output], [tokens], [defaults], [models.<key>]
           Auto-written on first run; tokens added by the TUI when you say so.

EXAMPLES
  # Simplest — uses the configured aspect ratio + auto-detected provider
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

  # Cheap, fast image via FLUX.2 [pro] on Replicate
  curds -model flux-2-pro -aspect-ratio 16:9 -image-resolution 2mp \
        -prompt "a watercolor fox in a meadow" -output /tmp/fox.png

  # 4K composite from two references via Nano Banana 2
  curds -model nano-banana-2 -image-resolution 4k \
        -input-image hero.png,logo.png \
        -prompt "hero shot with the logo on the bottle" -output /tmp/comp.png

  # Generate video with the default video model (Seedance 2.0 via Replicate):
  #   text-to-video, no input image required
  curds -prompt "a slow serene time-lapse of the milky way" -output /tmp/sky.mp4

  # Kling 3.0: image-to-video with an end frame, 1080p, 10s, lip-synced audio
  curds -model kling-v3 -input-image start.png -last-frame-image end.png \
        -prompt "the character turns and speaks to camera" \
        -video-resolution 1080p -video-duration 10 -output /tmp/kling.mp4

  # MiniMax H3 on Replicate: image-to-video from a first frame, 2K, 10s
  curds -model minimax-h3 -input-image still.png \
        -prompt "a smooth product turn with soft studio camera motion" \
        -video-resolution 2k -video-duration 10 -output /tmp/h3.mp4

  # Native xAI image-to-video with a reference image, 720p, 10s
  curds -provider xai -input-image still.png \
        -prompt "a smooth product turn with soft studio camera motion" \
        -video-resolution 720p -video-duration 10 -output /tmp/xai.mp4

  # Force the Replicate Grok 1.5 wrapper instead of native xAI
  curds -provider replicate -model grok-imagine-video-1.5 -input-image still.png \
        -prompt "a smooth product turn with soft studio camera motion" \
        -output /tmp/grok.mp4

  # Generate a Seedance 2.0 video via Replicate
  curds -provider replicate -model seedance-2 \
        -prompt "a cinematic 5 second shot of a glass sculpture forming" \
        -aspect-ratio 16:9 -video-duration 5 -output /tmp/seedance.mp4

  # Remove the background → transparent PNG (BRIA RMBG 2.0 on Replicate)
  curds -provider replicate -model remove-bg \
        -input-image photo.jpg -output cutout.png

  # Upscale 4x → PNG (Real-ESRGAN on Replicate)
  curds -provider replicate -model upscale \
        -input-image small.jpg -scale 4 -output big.png

  # Upscale 4x with Topaz instead (better on faces and text)
  curds -model upscale-pro -input-image small.jpg -scale 4 \
        -face-enhance -output big.png

  # Upscale a portrait with face enhancement
  curds -provider replicate -model upscale -face-enhance \
        -input-image headshot.jpg -output headshot-4x.png

FILES
  ~/.config/curds/config.toml    config (auto-created)
  ~/Desktop/curds/               default output directory (auto-created)
  .env                           optional; read from cwd

ENVIRONMENT
  CURDS_CONFIG                   override config file path
  OPENAI_API_KEY                 fallback OpenAI token
  REPLICATE_API_TOKEN            fallback Replicate token
  XAI_API_KEY                    fallback xAI token
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

	// If -output has an extension and -output-format wasn't explicitly set, derive
	// the format from the extension so the bytes match the filename.
	if opts.outputPath != "" && !flagWasSet("output-format") {
		if ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(opts.outputPath), ".")); ext != "" {
			switch ext {
			case "webp", "png", "mp4":
				opts.outputFormat = ext
			case "jpeg", "jpg":
				opts.outputFormat = "jpeg"
			}
		}
	}
	if !flagWasSet("model") && opts.outputFormat == "mp4" {
		opts.modelKey = cfg.DefaultVideoModel
	}
}

func applyModelOutputDefaults(opts *cliOptions, cfg *config.Config, model string, logger *logfmtLogger) {
	switch {
	case curds.IsVideoModel(model):
		if !flagWasSet("output-format") {
			opts.outputFormat = "mp4"
		}
		if !flagWasSet("aspect-ratio") {
			switch {
			case curds.IsGrokImagineVideoModel(model), curds.IsXaiVideoModel(model):
				opts.aspectRatio = "auto"
			case curds.IsMinimaxVideoModel(model), curds.IsKlingVideoModel(model),
				curds.IsSeedanceModel(model):
				// None of these accept "auto", and 1:1 is a poor video
				// default; landscape matches the models' own defaults.
				opts.aspectRatio = "16:9"
			}
		}
	case curds.IsSegmentationModel(model), curds.IsUpscaleModel(model):
		// bria/remove-background returns a transparent PNG and the upscalers
		// return an upscaled PNG. Forcing PNG here avoids saving with the
		// wrong extension when output.format is webp.
		if !flagWasSet("output-format") {
			opts.outputFormat = "png"
		}
	case curds.IsNanoBananaImageModel(model):
		// Nano Banana 2 emits jpg or png only, so webp — whether from config or
		// from an -output extension — would fail validation on every run.
		if !flagWasSet("output-format") && opts.outputFormat == "webp" {
			opts.outputFormat = "png"
		}
	default:
		return
	}
	if !flagWasSet("output") {
		if path, err := ensureDefaultOutputPath(cfg, opts.outputFormat); err == nil {
			opts.outputPath = path
		}
		return
	}
	// An explicit -output whose extension the model cannot emit (e.g.
	// -output cut.webp for remove-bg, which only returns PNG) would otherwise
	// save one format under another's name. Retarget the extension instead.
	if ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(opts.outputPath), ".")); ext != "" {
		if normalizeFormatExt(ext) != opts.outputFormat {
			opts.outputPath = strings.TrimSuffix(opts.outputPath, filepath.Ext(opts.outputPath)) + "." + opts.outputFormat
			if logger != nil {
				logger.info("output.extension_retargeted", "model", model, "format", opts.outputFormat, "path", opts.outputPath)
			}
		}
	}
}

// normalizeFormatExt maps a file extension onto curds' output-format spelling.
func normalizeFormatExt(ext string) string {
	if ext == "jpg" {
		return "jpeg"
	}
	return ext
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

func saveResult(opts *cliOptions, res *curds.Result) ([]string, error) {
	if res == nil {
		return nil, nil
	}
	if len(res.Videos) > 0 {
		return saveVideos(opts, res.Videos)
	}
	return saveImages(opts, res.Images)
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

func saveVideos(opts *cliOptions, videos []curds.Video) ([]string, error) {
	origExt := filepath.Ext(opts.outputPath)
	stem := strings.TrimSuffix(opts.outputPath, origExt)
	ext := origExt
	if ext == "" {
		ext = ".mp4"
	}

	paths := make([]string, 0, len(videos))
	for i, video := range videos {
		var path string
		switch {
		case len(videos) > 1:
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
		if err := os.WriteFile(path, video.Bytes, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		fmt.Println(path)
		paths = append(paths, path)
	}
	return paths, nil
}

// maybeStripAudio removes the audio track from each generated video file when
// -strip-audio is enabled. xAI/Grok always generate audio and the x.ai API has
// no mute option, so post-processing is the only way to get silent clips. Uses
// ffmpeg stream-copy (no re-encode). If ffmpeg is missing, it logs a skip and
// leaves the file untouched; a strip failure is logged but not fatal — the
// generated video is preserved either way. Events go to w as logfmt lines.
func maybeStripAudio(opts *cliOptions, videoCount int, paths []string, w io.Writer) {
	if !opts.stripAudio || videoCount == 0 {
		return
	}
	color := curds.IsTerminalWriter(w)
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		fmt.Fprint(w, curds.FormatLogLine("warn", "audio.strip_skipped",
			[]any{"reason", "ffmpeg not found on PATH", "hint", "install ffmpeg or pass -strip-audio=false"}, color))
		return
	}
	for _, p := range paths {
		if strings.ToLower(filepath.Ext(p)) != ".mp4" {
			continue
		}
		if err := stripAudioInPlace(ffmpeg, p); err != nil {
			fmt.Fprint(w, curds.FormatLogLine("error", "audio.strip_failed",
				[]any{"path", p, "err", err.Error()}, color))
			continue
		}
		fmt.Fprint(w, curds.FormatLogLine("info", "audio.stripped", []any{"path", p}, color))
	}
}

// stripAudioInPlace rewrites path with its audio track removed, copying the
// remaining streams without re-encoding. It writes to a sibling temp file and
// atomically renames over the original so a failure never corrupts the result.
func stripAudioInPlace(ffmpeg, path string) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), "curds-noaudio-*.mp4")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	_ = tmp.Close()
	// -c copy: stream-copy (no re-encode); -an: drop all audio streams.
	cmd := exec.Command(ffmpeg, "-y", "-loglevel", "error", "-i", path, "-c", "copy", "-an", tmpName)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("ffmpeg: %v: %s", err, strings.TrimSpace(string(out)))
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	return nil
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
	case "xai":
		return "XAI_API_KEY"
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
