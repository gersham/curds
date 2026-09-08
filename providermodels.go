package curds

import (
	"fmt"
	"strings"
)

// providerModelSpec is one row of the hand-maintained provider/model table.
// Default is the resolved identifier sent upstream when the caller omits
// -model. Supported is the user-facing list shown in mismatch errors
// (config keys / canonical ids). aliases are extra identifiers that count
// as compatible (resolved names, fast variants) and are not repeated in
// the error list.
type providerModelSpec struct {
	Default   string
	Supported []string
	aliases   []string
}

// providerModels is the single table of per-provider defaults and supported
// models. Keep in sync with config.builtinModels and the PROVIDERS help text.
var providerModels = map[string]providerModelSpec{
	ProviderOpenAI: {
		Default: DefaultOpenAIModel,
		Supported: []string{
			"gpt-image-2.5",
			GPTImage2Model,
		},
		aliases: []string{
			GPTImage25Model,
		},
	},
	ProviderXai: {
		Default:   DefaultXaiVideoModel,
		Supported: []string{DefaultXaiVideoModel},
	},
	ProviderReplicate: {
		Default: DefaultReplicateModel,
		Supported: []string{
			"gpt-image-2",
			"flux-2-pro",
			"nano-banana-2",
			"seedance-2",
			"kling-v3",
			"minimax-h3",
			"grok-imagine-video-1.5",
			"remove-bg",
			"upscale",
			"upscale-pro",
		},
		aliases: []string{
			DefaultReplicateModel,
			FluxImageModel,
			NanoBananaImageModel,
			SeedanceVideoModel,
			"bytedance/seedance-2.0-fast",
			KlingVideoModel,
			MinimaxVideoModel,
			GrokImagineVideoModel,
			DefaultSegmentationModel,
			DefaultUpscaleModel,
			TopazUpscaleModel,
		},
	},
}

// DefaultModel returns the provider's default model name.
func DefaultModel(provider string) string {
	if spec, ok := providerModels[provider]; ok {
		return spec.Default
	}
	return ""
}

// SelectDefaultModel returns current when it is already compatible with
// provider, otherwise the provider's default. Used when -provider is set
// and -model is not: keep a compatible video default (replicate + mp4 →
// seedance-2) but replace a global image default the provider cannot run
// (xai + gpt-image-2 → grok-imagine-video).
func SelectDefaultModel(provider, current string) string {
	if strings.TrimSpace(current) != "" && modelCompatible(provider, current) {
		return current
	}
	return DefaultModel(provider)
}

// CheckProviderModel reports whether model can run on provider. An empty
// provider or model is skipped so callers can validate incrementally.
// Incompatible pairs fail locally with the supported list; no network call
// should follow.
func CheckProviderModel(provider, model string) error {
	if provider == "" || strings.TrimSpace(model) == "" {
		return nil
	}
	spec, ok := providerModels[provider]
	if !ok {
		return nil
	}
	if modelCompatible(provider, model) {
		return nil
	}
	return fmt.Errorf("provider %s does not support model %q (supported: %s)",
		provider, model, strings.Join(spec.Supported, ", "))
}

func modelCompatible(provider, model string) bool {
	spec, ok := providerModels[provider]
	if !ok {
		return false
	}
	m := strings.TrimSpace(strings.ToLower(model))
	if m == "" {
		return false
	}
	for _, name := range spec.matchNames() {
		if modelMatches(name, m) {
			return true
		}
	}
	// Replicate accepts any owner/name identifier (documented passthrough
	// for models curds does not wrap first-class).
	if provider == ProviderReplicate && replicateOwnerName(m) {
		return true
	}
	return false
}

func (s providerModelSpec) matchNames() []string {
	out := make([]string, 0, 1+len(s.Supported)+len(s.aliases))
	if s.Default != "" {
		out = append(out, s.Default)
	}
	out = append(out, s.Supported...)
	out = append(out, s.aliases...)
	return out
}

func modelMatches(name, model string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	if model == name {
		return true
	}
	// Version pin of a Replicate-style owner/name identifier.
	if strings.Contains(name, "/") && strings.HasPrefix(model, name+":") && strings.TrimSpace(model[len(name)+1:]) != "" {
		return true
	}
	return false
}

func replicateOwnerName(model string) bool {
	base := model
	if i := strings.Index(model, ":"); i >= 0 {
		base = model[:i]
	}
	owner, name, ok := strings.Cut(base, "/")
	return ok && owner != "" && name != "" && !strings.Contains(name, "/")
}
