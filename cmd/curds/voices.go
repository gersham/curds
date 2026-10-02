package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gersham/curds"
	"github.com/gersham/curds/config"
)

// newVoicesProvider builds the ElevenLabs client `curds voices` talks to. It is
// a var so tests can point it at an httptest server, like newRunProvider.
var newVoicesProvider = func() *curds.ElevenLabsProvider { return &curds.ElevenLabsProvider{} }

// voicesOptions is the `curds voices` flag surface.
type voicesOptions struct {
	search  string
	accent  string
	gender  string
	age     string
	source  string
	limit   int
	token   string
	timeout time.Duration
}

// voicesHelpText returns the `curds voices` help.
func voicesHelpText() string {
	return `NAME
  curds voices — find ElevenLabs voices for -model tts-elevenlabs

SYNOPSIS
  curds voices [-search TEXT] [-accent ACCENT] [-gender GENDER] [-age AGE]
               [-source all|account|library] [-limit N]

DESCRIPTION
  Lists the voices your ElevenLabs account can use (GET /v2/voices: your own,
  saved, and premade voices) and searches the shared voice library
  (GET /v1/shared-voices), one voice per line on stdout:
    source=account|library id=... name=... accent=... age=... gender=...
    description=...
  Pass an id to -voice on the direct ElevenLabs route
  (curds -model tts-elevenlabs -voice ID). Account voices also resolve by
  name. Filters are applied by ElevenLabs on each voice's labels. Needs an
  elevenlabs token.

FLAGS
  -search TEXT          search names, descriptions, and labels
  -accent ACCENT        e.g. british, american, irish, australian
  -gender GENDER        e.g. female, male, neutral
  -age AGE              e.g. young, middle_aged, old
  -source VALUE         all (default), account, or library
  -limit N              voices per source, 1-100 (default 30)
  -token STRING         ElevenLabs API key (overrides config/.env/env)
  -timeout DURATION     overall timeout (default 1m, 0 disables)

TOKEN RESOLUTION (first non-empty wins)
  1. -token flag
  2. ~/.config/curds/config.toml [tokens].elevenlabs
  3. .env in cwd
  4. ELEVENLABS_API_KEY

EXIT CODES
  0  success
  1  upstream error
  2  invalid flag or argument

EXAMPLES
  # A British female narrator from the library or your account
  curds voices -search narrator -accent british -gender female

  # Only the voices already in your account
  curds voices -source account
`
}

// realMainVoices implements `curds voices`. Voice lines go to stdout; logs go
// to stderr as logfmt, like the rest of curds.
func realMainVoices(logger *logfmtLogger, start time.Time, args []string) error {
	fs := flag.NewFlagSet("curds voices", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	opts := &voicesOptions{}
	fs.StringVar(&opts.search, "search", "", "Search names, descriptions, and labels")
	fs.StringVar(&opts.accent, "accent", "", "Accent label filter (e.g. british)")
	fs.StringVar(&opts.gender, "gender", "", "Gender label filter (e.g. female)")
	fs.StringVar(&opts.age, "age", "", "Age label filter (e.g. middle_aged)")
	fs.StringVar(&opts.source, "source", "all", "Voices to list: all, account, or library")
	fs.IntVar(&opts.limit, "limit", 30, "Voices per source, 1-100")
	fs.StringVar(&opts.token, "token", "", "ElevenLabs API key (overrides config/.env/env)")
	fs.DurationVar(&opts.timeout, "timeout", time.Minute, "Overall timeout (0 disables)")
	fs.Usage = func() { fmt.Fprint(fs.Output(), voicesHelpText()) }

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(os.Stderr, voicesHelpText())
			return nil
		}
		return &usageError{err: err}
	}
	if fs.NArg() > 0 {
		return &usageError{err: fmt.Errorf("curds voices takes no arguments, got %q; filter with -search TEXT", fs.Arg(0))}
	}
	switch opts.source {
	case "all", "account", "library":
	default:
		return &usageError{err: fmt.Errorf("-source must be all, account, or library, got %q", opts.source)}
	}
	if opts.limit < 1 || opts.limit > 100 {
		return &usageError{err: fmt.Errorf("-limit must be 1-100, got %d", opts.limit)}
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
		token = config.ResolveToken(curds.ProviderElevenLabs, cfg, dotenv, os.Getenv)
	}
	if token == "" {
		return fmt.Errorf("no elevenlabs token available; set it in %s, .env, or ELEVENLABS_API_KEY", cfg.Path)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if opts.timeout > 0 {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, opts.timeout)
		defer stop()
	}

	provider := newVoicesProvider()
	q := curds.VoiceQuery{Search: opts.search, Accent: opts.accent, Gender: opts.gender, Age: opts.age, Limit: opts.limit}
	counts := map[string]int{}
	for _, src := range []string{"account", "library"} {
		if opts.source != "all" && opts.source != src {
			continue
		}
		list := provider.AccountVoices
		if src == "library" {
			list = provider.LibraryVoices
		}
		voices, err := list(ctx, token, q)
		if err != nil {
			return fmt.Errorf("list %s voices: %w", src, err)
		}
		for _, v := range voices {
			fmt.Println(v.String())
		}
		counts[src] = len(voices)
		logger.info("voices.listed", "source", src, "count", len(voices))
	}
	logger.info("voices.completed",
		"account", counts["account"],
		"library", counts["library"],
		"duration_ms", time.Since(start).Milliseconds(),
	)
	return nil
}
