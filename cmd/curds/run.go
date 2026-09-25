package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/gersham/curds"
	"github.com/gersham/curds/config"
)

// newRunProvider builds the Replicate client `curds run` talks to. It is a var
// so tests can point it at an httptest server — the same reason Client keeps
// its providers pluggable.
var newRunProvider = func() *curds.ReplicateProvider { return &curds.ReplicateProvider{} }

// runOptions is the `curds run` flag surface. Flags come first, then the
// positional OWNER/MODEL[:VERSION] and key=value pairs — stdlib flag style.
type runOptions struct {
	outputPath   string
	schema       bool
	token        string
	pollInterval time.Duration
	timeout      time.Duration
	verbose      bool
	jsonOutput   bool
}

// runHelpText returns the `curds run` help: a passthrough for any Replicate
// model, with no field mapping of its own.
func runHelpText() string {
	return `NAME
  curds run — run any Replicate model with raw key=value inputs

SYNOPSIS
  curds run [flags] OWNER/MODEL[:VERSION] [key=value ...]
  curds run -schema OWNER/MODEL

DESCRIPTION
  Creates one Replicate prediction with the inputs you pass and nothing else —
  no field mapping, no defaults beyond the model's own — then polls it to
  completion and downloads the output. Use it for models curds does not wrap
  first-class (GET the schema with -schema to see what a model accepts).
  Values are sent as JSON: ` + "`key=@file.png`" + ` uploads a local file as a data
  URL, ` + "`key=5`" + ` / ` + "`key=true`" + ` / ` + "`key=[1,2]`" + ` are sent as that JSON value,
  and anything unparseable is sent as a plain string. Repeat a key to send an
  array: ` + "`ref=@a.png ref=@b.png`" + `.

FLAGS
  -output PATH          output file path for downloaded assets
                        default: <output.directory>/<unix_milli>.<ext>, where
                        the extension comes from the output URL (bin if the
                        URL carries none). With several outputs the paths get
                        -1, -2, … before the extension.
  -schema               print the model's inputs (name, type, default, enum,
                        min/max, description), one per line, then exit. No
                        prediction is created.
  -json                 print the final prediction JSON to stdout instead of
                        downloading. Applied automatically when the output
                        carries no URLs (then the output JSON is printed).
  -token STRING         Replicate API token (overrides config/.env/env)
  -poll-interval DURATION  status poll cadence (default 2s)
  -timeout DURATION     overall timeout (default 10m, 0 disables)
  -verbose              include debug-level logs on stderr

TOKEN RESOLUTION (first non-empty wins)
  1. -token flag
  2. ~/.config/curds/config.toml [tokens].replicate
  3. .env in cwd
  4. REPLICATE_API_TOKEN

EXIT CODES
  0  success
  1  upstream error or prediction failure
  2  invalid flag, model reference, or input argument

EXAMPLES
  # What does this model take?
  curds run -schema sync/lipsync-2-pro

  # Text-to-video on a first-class model, without curds' field mapping
  curds run bytedance/seedance-2.0 prompt="a city at dusk" duration=5

  # Upload two local reference images and read the output URL
  curds run -json owner/model prompt="blend these" ref=@a.png ref=@b.png

  # Pin a version and name the output file
  curds run -output /tmp/out.mp4 owner/model:abc123 prompt="a slow pan"
`
}

// realMainRun implements `curds run`. Output path(s) go to stdout; every stage
// logs logfmt to stderr, like the rest of curds.
func realMainRun(logger *logfmtLogger, start time.Time, args []string) error {
	fs := flag.NewFlagSet("curds run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	opts := &runOptions{}
	fs.StringVar(&opts.outputPath, "output", "", "Output file path (default: <output.directory>/<unix_milli>.<ext>)")
	fs.BoolVar(&opts.schema, "schema", false, "Print the model's input schema and exit (no prediction)")
	fs.BoolVar(&opts.jsonOutput, "json", false, "Print the final prediction JSON instead of downloading output")
	fs.StringVar(&opts.token, "token", "", "Replicate API token (overrides config/.env/env)")
	fs.DurationVar(&opts.pollInterval, "poll-interval", 2*time.Second, "Polling interval for replicate")
	fs.DurationVar(&opts.timeout, "timeout", 10*time.Minute, "Overall timeout (0 disables)")
	fs.BoolVar(&opts.verbose, "verbose", false, "Verbose debug logs to stderr")
	fs.Usage = func() { fmt.Fprint(fs.Output(), runHelpText()) }

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(os.Stderr, runHelpText())
			return nil
		}
		return &usageError{err: err}
	}

	rest := fs.Args()
	if len(rest) == 0 {
		return &usageError{err: errors.New("usage: curds run [flags] OWNER/MODEL[:VERSION] [key=value ...] (see curds run -h)")}
	}
	model, pairs := rest[0], rest[1:]
	if !isReplicateModelRef(model) {
		return &usageError{err: fmt.Errorf("model %q must be an owner/name Replicate reference (curds run is Replicate-only)", model)}
	}

	cfg, _, err := config.LoadOrCreate()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	dotenv, err := config.LoadDotEnv(".env")
	if err != nil {
		logger.error("dotenv.read_failed", "err", err.Error())
	}

	token := opts.token
	if token == "" {
		token = config.ResolveToken(curds.ProviderReplicate, cfg, dotenv, os.Getenv)
	}
	if token == "" {
		return fmt.Errorf("no replicate token available; set it in %s, .env, or REPLICATE_API_TOKEN", cfg.Path)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if opts.timeout > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, opts.timeout)
		defer stop()
	}

	provider := newRunProvider()
	if opts.schema {
		if len(pairs) > 0 {
			return &usageError{err: errors.New("-schema takes no input arguments")}
		}
		fields, err := provider.Schema(ctx, token, model)
		if err != nil {
			return err
		}
		logger.info("run.schema", "model", model, "fields", len(fields))
		for _, field := range fields {
			fmt.Println(field.String())
		}
		return nil
	}

	input, err := curds.ParseRunInput(pairs)
	if err != nil {
		return &usageError{err: err}
	}

	req := &curds.Request{
		Provider:     curds.ProviderReplicate,
		Token:        token,
		Model:        model,
		PollInterval: opts.pollInterval,
		Logger:       os.Stderr,
		Verbose:      opts.verbose,
	}
	logger.info("run.started", "model", model, "inputs", len(input))

	res, err := provider.Run(ctx, req, input)
	if err != nil {
		return err
	}
	logger.info("run.succeeded", "id", res.ID, "output_urls", len(res.URLs))

	// No URLs (a text/blob output) means there is nothing to download: print
	// the JSON so the caller still gets the result, same as -json.
	if opts.jsonOutput || len(res.URLs) == 0 {
		raw := res.Prediction
		if !opts.jsonOutput || len(raw) == 0 {
			raw = res.Output
		}
		fmt.Println(string(raw))
		return nil
	}

	paths, err := saveRunOutputs(ctx, provider, token, opts, cfg, res.URLs)
	if err != nil {
		return err
	}
	logger.info("run.completed",
		"id", res.ID,
		"outputs", len(paths),
		"duration_ms", time.Since(start).Milliseconds(),
		"paths", strings.Join(paths, ","),
	)
	return nil
}

// isReplicateModelRef reports whether ref looks like Replicate's owner/name
// (optionally with a :version pin). Deeper validation is Replicate's job.
func isReplicateModelRef(ref string) bool {
	base, _, _ := strings.Cut(ref, ":")
	owner, name, ok := strings.Cut(base, "/")
	return ok && owner != "" && name != "" && !strings.Contains(name, "/")
}

// saveRunOutputs downloads every output URL and writes it to disk, printing
// each saved path to stdout. opts.outputPath is used as-is for a single output;
// with several, each gets -1, -2, … before the extension.
func saveRunOutputs(ctx context.Context, provider *curds.ReplicateProvider, token string, opts *runOptions, cfg *config.Config, urls []string) ([]string, error) {
	target := opts.outputPath
	if target == "" {
		target = filepath.Join(
			config.ExpandTilde(cfg.Output.Directory),
			fmt.Sprintf("%d.%s", time.Now().UnixMilli(), runOutputExt(urls)),
		)
	}
	ext := filepath.Ext(target)
	stem := strings.TrimSuffix(target, ext)

	paths := make([]string, 0, len(urls))
	for i, u := range urls {
		path := target
		if len(urls) > 1 {
			path = fmt.Sprintf("%s-%d%s", stem, i+1, ext)
		}
		body, err := provider.Download(ctx, token, u)
		if err != nil {
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		if dir := filepath.Dir(path); dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, err
			}
		}
		if err := os.WriteFile(path, body, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		fmt.Println(path)
		paths = append(paths, path)
	}
	return paths, nil
}

// runOutputExt picks the default output extension from the first output URL
// that carries a filename, falling back to "bin" for opaque or extensionless
// URLs.
func runOutputExt(urls []string) string {
	for _, raw := range urls {
		u, err := url.Parse(raw)
		if err != nil {
			continue
		}
		if ext := strings.TrimPrefix(filepath.Ext(u.Path), "."); ext != "" {
			return ext
		}
	}
	return "bin"
}
