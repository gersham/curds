// Package config loads and manages the curds TOML configuration.
//
// Configuration lives at $XDG_CONFIG_HOME/curds/config.toml (or
// ~/.config/curds/config.toml). The file is auto-created with defaults the
// first time curds runs.
package config

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

// DefaultTOML is the literal content written when no config file exists.
const DefaultTOML = `# Curds image-generation config.
#
# Resolution priority:
#   API tokens:   config (this file) > .env in cwd > process env vars
#   Model params: CLI flag > config defaults

# Provider: "openai", "replicate", or "" to auto-detect from available tokens.
provider = ""

# Default model key (looked up in [models.<key>] below).
default_model = "gpt-image-2"

# Output settings.
[output]
# Files land here. Tilde expansion is supported.
directory = "~/Desktop/curds"
# Format used when -o has no extension and the model supports it.
format = "webp"
# Compression for webp/jpeg, 0-100 (OpenAI only).
compression = 90

# API tokens. Leave blank to defer to .env / env vars.
[tokens]
openai = ""
replicate = ""

# Default image-gen parameters. Override via CLI flags.
[defaults]
quality = "auto"
aspect_ratio = "1:1"
background = "auto"
moderation = "auto"
number_of_images = 1

# Models. Each entry maps a logical key (used with -model) to the
# provider-specific model identifier. Add more as you go.
[models.gpt-image-2]
openai_name = "gpt-image-2"
replicate_name = "openai/gpt-image-2"
`

// Config is the parsed config file.
type Config struct {
	Provider     string                 `toml:"provider"`
	DefaultModel string                 `toml:"default_model"`
	Output       OutputConfig           `toml:"output"`
	Tokens       TokensConfig           `toml:"tokens"`
	Defaults     DefaultsConfig         `toml:"defaults"`
	Models       map[string]ModelConfig `toml:"models"`

	// Path is the file the config was read from (for diagnostics).
	Path string `toml:"-"`
}

type OutputConfig struct {
	Directory   string `toml:"directory"`
	Format      string `toml:"format"`
	Compression int    `toml:"compression"`
}

type TokensConfig struct {
	OpenAI    string `toml:"openai"`
	Replicate string `toml:"replicate"`
}

type DefaultsConfig struct {
	Quality        string `toml:"quality"`
	AspectRatio    string `toml:"aspect_ratio"`
	Background     string `toml:"background"`
	Moderation     string `toml:"moderation"`
	NumberOfImages int    `toml:"number_of_images"`
}

type ModelConfig struct {
	OpenAIName    string `toml:"openai_name"`
	ReplicateName string `toml:"replicate_name"`
}

// LoadOrCreate finds the config file, creating it with defaults if missing,
// then parses it.
func LoadOrCreate() (*Config, bool, error) {
	path, err := DefaultPath()
	if err != nil {
		return nil, false, err
	}
	return LoadOrCreateAt(path)
}

// LoadOrCreateAt is the same as LoadOrCreate but takes an explicit path.
// Returns the config, whether the file was created on this call, and any
// error.
func LoadOrCreateAt(path string) (*Config, bool, error) {
	created := false
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			return nil, false, fmt.Errorf("create config dir: %w", err)
		}
		if err := os.WriteFile(path, []byte(DefaultTOML), 0o600); err != nil {
			return nil, false, fmt.Errorf("write default config: %w", err)
		}
		created = true
	} else if err != nil {
		return nil, false, fmt.Errorf("stat config: %w", err)
	}

	var cfg Config
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, false, fmt.Errorf("parse %s: %w", path, err)
	}
	cfg.Path = path
	cfg.applyZeroDefaults()
	return &cfg, created, nil
}

// applyZeroDefaults fills in sensible values when a field is zero so the
// rest of the program can rely on them.
func (c *Config) applyZeroDefaults() {
	if c.DefaultModel == "" {
		c.DefaultModel = "gpt-image-2"
	}
	if c.Output.Directory == "" {
		c.Output.Directory = "~/Desktop/curds"
	}
	if c.Output.Format == "" {
		c.Output.Format = "webp"
	}
	if c.Output.Compression == 0 {
		c.Output.Compression = 90
	}
	if c.Defaults.Quality == "" {
		c.Defaults.Quality = "auto"
	}
	if c.Defaults.AspectRatio == "" {
		c.Defaults.AspectRatio = "1:1"
	}
	if c.Defaults.Background == "" {
		c.Defaults.Background = "auto"
	}
	if c.Defaults.Moderation == "" {
		c.Defaults.Moderation = "auto"
	}
	if c.Defaults.NumberOfImages == 0 {
		c.Defaults.NumberOfImages = 1
	}
	if c.Models == nil {
		c.Models = map[string]ModelConfig{}
	}
	if _, ok := c.Models["gpt-image-2"]; !ok {
		c.Models["gpt-image-2"] = ModelConfig{
			OpenAIName:    "gpt-image-2",
			ReplicateName: "openai/gpt-image-2",
		}
	}
}

// DefaultPath returns the canonical config path.
func DefaultPath() (string, error) {
	if env := os.Getenv("CURDS_CONFIG"); env != "" {
		return env, nil
	}
	if x := os.Getenv("XDG_CONFIG_HOME"); x != "" {
		return filepath.Join(x, "curds", "config.toml"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "curds", "config.toml"), nil
}

// Save writes the entire config back to disk. Used by the interactive
// settings form and by the one-shot token capture flow. The file is
// truncated and re-marshalled via the toml encoder.
func (c *Config) Save() error {
	if c.Path == "" {
		return errors.New("config has no source path")
	}
	f, err := os.OpenFile(c.Path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := toml.NewEncoder(f)
	return enc.Encode(c)
}

// SaveTokens is a deprecated alias kept for backwards compatibility.
// New code should call Save directly.
func (c *Config) SaveTokens() error { return c.Save() }

// LoadDotEnv reads a KEY=value file and returns a map of the assignments.
// Lines starting with # and blank lines are ignored. A leading "export "
// is stripped. Surrounding single or double quotes around the value are
// stripped. Missing files return (nil, nil).
func LoadDotEnv(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	out := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		idx := strings.Index(line, "=")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		if len(val) >= 2 {
			if (val[0] == '"' && val[len(val)-1] == '"') ||
				(val[0] == '\'' && val[len(val)-1] == '\'') {
				val = val[1 : len(val)-1]
			}
		}
		out[key] = val
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// ResolveToken applies the config > .env > env-var precedence.
func ResolveToken(provider string, cfg *Config, dotenv map[string]string, getenv func(string) string) string {
	var fromCfg, envName string
	switch provider {
	case "openai":
		fromCfg = cfg.Tokens.OpenAI
		envName = "OPENAI_API_KEY"
	case "replicate":
		fromCfg = cfg.Tokens.Replicate
		envName = "REPLICATE_API_TOKEN"
	default:
		return ""
	}
	if strings.TrimSpace(fromCfg) != "" {
		return fromCfg
	}
	if v, ok := dotenv[envName]; ok && strings.TrimSpace(v) != "" {
		return v
	}
	if getenv != nil {
		return getenv(envName)
	}
	return ""
}

// DetectProvider returns the first provider with a non-empty token. The
// configured provider wins outright; otherwise OpenAI is preferred over
// Replicate.
func DetectProvider(cfg *Config, dotenv map[string]string, getenv func(string) string) string {
	if cfg.Provider != "" {
		return cfg.Provider
	}
	if ResolveToken("openai", cfg, dotenv, getenv) != "" {
		return "openai"
	}
	if ResolveToken("replicate", cfg, dotenv, getenv) != "" {
		return "replicate"
	}
	return ""
}

// ExpandTilde replaces a leading "~/" with the user's home directory.
func ExpandTilde(p string) string {
	if !strings.HasPrefix(p, "~/") && p != "~" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	if p == "~" {
		return home
	}
	return filepath.Join(home, p[2:])
}

// ResolveModel returns the provider-specific model identifier for a logical
// model key.
func ResolveModel(cfg *Config, key, provider string) string {
	if key == "" {
		key = cfg.DefaultModel
	}
	m, ok := cfg.Models[key]
	if !ok {
		// Caller passed a raw provider model name; just use it.
		return key
	}
	switch provider {
	case "openai":
		if m.OpenAIName != "" {
			return m.OpenAIName
		}
		return key
	case "replicate":
		if m.ReplicateName != "" {
			return m.ReplicateName
		}
		return key
	}
	return key
}
