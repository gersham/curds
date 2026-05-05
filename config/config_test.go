package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadOrCreateAtCreatesDefault(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "config.toml")

	cfg, created, err := LoadOrCreateAt(path)
	if err != nil {
		t.Fatal(err)
	}
	if !created {
		t.Fatal("expected created=true on first run")
	}
	if cfg.DefaultModel != "gpt-image-2" {
		t.Errorf("DefaultModel: %q", cfg.DefaultModel)
	}
	if cfg.Output.Directory == "" {
		t.Errorf("Output.Directory empty")
	}
	if cfg.Output.Compression != 90 {
		t.Errorf("Output.Compression: %d", cfg.Output.Compression)
	}
	if _, ok := cfg.Models["gpt-image-2"]; !ok {
		t.Errorf("default model missing from Models map")
	}
	if cfg.Models["gpt-image-2"].ReplicateName != "openai/gpt-image-2" {
		t.Errorf("replicate name: %q", cfg.Models["gpt-image-2"].ReplicateName)
	}
	if cfg.Models["seedance-2"].ReplicateName != "bytedance/seedance-2.0" {
		t.Errorf("seedance replicate name: %q", cfg.Models["seedance-2"].ReplicateName)
	}
	if cfg.Models["remove-bg"].ReplicateName != "bria/remove-background" {
		t.Errorf("remove-bg replicate name: %q", cfg.Models["remove-bg"].ReplicateName)
	}

	// Second call should NOT recreate.
	_, created2, err := LoadOrCreateAt(path)
	if err != nil {
		t.Fatal(err)
	}
	if created2 {
		t.Fatal("expected created=false on second run")
	}
}

func TestLoadDotEnv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".env")
	body := `
# comment
OPENAI_API_KEY=sk-abc
export REPLICATE_API_TOKEN="r8_xyz"
QUOTED='single quoted'
EMPTY=
WITH_COMMENT=value # not stripped
`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	env, err := LoadDotEnv(path)
	if err != nil {
		t.Fatal(err)
	}
	if env["OPENAI_API_KEY"] != "sk-abc" {
		t.Errorf("OPENAI_API_KEY: %q", env["OPENAI_API_KEY"])
	}
	if env["REPLICATE_API_TOKEN"] != "r8_xyz" {
		t.Errorf("REPLICATE_API_TOKEN: %q", env["REPLICATE_API_TOKEN"])
	}
	if env["QUOTED"] != "single quoted" {
		t.Errorf("QUOTED: %q", env["QUOTED"])
	}
	if env["EMPTY"] != "" {
		t.Errorf("EMPTY: %q", env["EMPTY"])
	}
}

func TestLoadDotEnvMissing(t *testing.T) {
	env, err := LoadDotEnv(filepath.Join(t.TempDir(), "no-such"))
	if err != nil {
		t.Fatal(err)
	}
	if env != nil {
		t.Fatal("expected nil for missing file")
	}
}

func TestResolveTokenPriority(t *testing.T) {
	cfg := &Config{Tokens: TokensConfig{OpenAI: "from-config"}}
	dotenv := map[string]string{"OPENAI_API_KEY": "from-env-file"}
	getenv := func(k string) string {
		if k == "OPENAI_API_KEY" {
			return "from-process-env"
		}
		return ""
	}

	// config wins
	if got := ResolveToken("openai", cfg, dotenv, getenv); got != "from-config" {
		t.Errorf("expected config, got %q", got)
	}
	// .env beats process env
	cfg.Tokens.OpenAI = ""
	if got := ResolveToken("openai", cfg, dotenv, getenv); got != "from-env-file" {
		t.Errorf("expected dotenv, got %q", got)
	}
	// process env as last resort
	delete(dotenv, "OPENAI_API_KEY")
	if got := ResolveToken("openai", cfg, dotenv, getenv); got != "from-process-env" {
		t.Errorf("expected process env, got %q", got)
	}
	// empty everywhere
	if got := ResolveToken("openai", cfg, nil, func(string) string { return "" }); got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestDetectProviderPrefersConfig(t *testing.T) {
	cfg := &Config{Provider: "replicate", Tokens: TokensConfig{OpenAI: "x"}}
	if got := DetectProvider(cfg, nil, func(string) string { return "" }); got != "replicate" {
		t.Errorf("config provider should win, got %q", got)
	}
}

func TestDetectProviderPrefersOpenAI(t *testing.T) {
	cfg := &Config{Tokens: TokensConfig{OpenAI: "x", Replicate: "y"}}
	if got := DetectProvider(cfg, nil, func(string) string { return "" }); got != "openai" {
		t.Errorf("expected openai, got %q", got)
	}
}

func TestExpandTilde(t *testing.T) {
	home, _ := os.UserHomeDir()
	cases := map[string]string{
		"~/Desktop/curds": filepath.Join(home, "Desktop", "curds"),
		"~":               home,
		"/tmp/x":          "/tmp/x",
		"./relative":      "./relative",
	}
	for in, want := range cases {
		if got := ExpandTilde(in); got != want {
			t.Errorf("ExpandTilde(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestResolveModel(t *testing.T) {
	cfg := &Config{
		DefaultModel: "gpt-image-2",
		Models: map[string]ModelConfig{
			"gpt-image-2": {OpenAIName: "gpt-image-2", ReplicateName: "openai/gpt-image-2"},
			"seedance-2":  {ReplicateName: "bytedance/seedance-2.0"},
		},
	}
	if got := ResolveModel(cfg, "", "openai"); got != "gpt-image-2" {
		t.Errorf("openai default: %q", got)
	}
	if got := ResolveModel(cfg, "", "replicate"); got != "openai/gpt-image-2" {
		t.Errorf("replicate default: %q", got)
	}
	// Unknown key: passes through.
	if got := ResolveModel(cfg, "owner/custom-model:abc", "replicate"); got != "owner/custom-model:abc" {
		t.Errorf("passthrough: %q", got)
	}
	if got := ResolveModel(cfg, "seedance-2", "replicate"); got != "bytedance/seedance-2.0" {
		t.Errorf("seedance: %q", got)
	}
}

func TestDefaultTOMLParses(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")
	if err := os.WriteFile(path, []byte(DefaultTOML), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := LoadOrCreateAt(path)
	if err != nil {
		t.Fatal(err)
	}
	// Sanity-check that the embedded TOML survives a round-trip.
	if !strings.Contains(DefaultTOML, "[models.gpt-image-2]") {
		t.Errorf("default TOML missing models section")
	}
	if !strings.Contains(DefaultTOML, "[models.seedance-2]") {
		t.Errorf("default TOML missing seedance model")
	}
	if cfg.Output.Directory == "" {
		t.Errorf("output directory missing after parse")
	}
}
