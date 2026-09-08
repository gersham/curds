package curds

import (
	"context"
	"strings"
	"testing"
)

func TestDefaultModelFromTable(t *testing.T) {
	cases := map[string]string{
		ProviderOpenAI:    DefaultOpenAIModel,
		ProviderReplicate: DefaultReplicateModel,
		ProviderXai:       DefaultXaiVideoModel,
		"unknown":         "",
	}
	for provider, want := range cases {
		if got := DefaultModel(provider); got != want {
			t.Errorf("DefaultModel(%q) = %q, want %q", provider, got, want)
		}
	}
}

func TestSelectDefaultModel(t *testing.T) {
	cases := []struct {
		name     string
		provider string
		current  string
		want     string
	}{
		{"xai replaces openai image default", ProviderXai, "gpt-image-2", DefaultXaiVideoModel},
		{"replicate replaces gpt-image-2.5", ProviderReplicate, "gpt-image-2.5", DefaultReplicateModel},
		{"xai replaces replicate video default", ProviderXai, "seedance-2", DefaultXaiVideoModel},
		{"xai empty uses its default", ProviderXai, "", DefaultXaiVideoModel},
		{"openai keeps its image default", ProviderOpenAI, "gpt-image-2", "gpt-image-2"},
		{"openai keeps the 2.5 key", ProviderOpenAI, "gpt-image-2.5", "gpt-image-2.5"},
		{"openai keeps the resolved 2.5 id", ProviderOpenAI, GPTImage25Model, GPTImage25Model},
		{"openai empty uses its default", ProviderOpenAI, "", DefaultOpenAIModel},
		{"replicate keeps image default key", ProviderReplicate, "gpt-image-2", "gpt-image-2"},
		{"replicate keeps mp4 video default", ProviderReplicate, "seedance-2", "seedance-2"},
		{"replicate empty uses its default", ProviderReplicate, "", DefaultReplicateModel},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := SelectDefaultModel(tc.provider, tc.current)
			if got != tc.want {
				t.Fatalf("SelectDefaultModel(%q, %q) = %q, want %q", tc.provider, tc.current, got, tc.want)
			}
		})
	}
}

func TestCheckProviderModel(t *testing.T) {
	t.Run("incompatible pair names provider, model, and supported list", func(t *testing.T) {
		err := CheckProviderModel(ProviderXai, "gpt-image-2")
		if err == nil {
			t.Fatal("expected error")
		}
		msg := err.Error()
		for _, want := range []string{ProviderXai, "gpt-image-2", DefaultXaiVideoModel, "supported:"} {
			if !strings.Contains(msg, want) {
				t.Errorf("error %q does not contain %q", msg, want)
			}
		}
	})
	t.Run("openai rejects replicate video key", func(t *testing.T) {
		err := CheckProviderModel(ProviderOpenAI, "seedance-2")
		if err == nil {
			t.Fatal("expected error")
		}
		msg := err.Error()
		if !strings.Contains(msg, ProviderOpenAI) || !strings.Contains(msg, "seedance-2") || !strings.Contains(msg, "gpt-image-2.5") {
			t.Errorf("error %q missing provider/model/supported", msg)
		}
	})
	t.Run("compatible pairs pass through", func(t *testing.T) {
		pairs := [][2]string{
			{ProviderOpenAI, "gpt-image-2"},
			{ProviderOpenAI, "gpt-image-2.5"},
			{ProviderOpenAI, GPTImage25Model},
			{ProviderXai, "grok-imagine-video"},
			{ProviderReplicate, "gpt-image-2"},
			{ProviderReplicate, "openai/gpt-image-2"},
			{ProviderReplicate, "seedance-2"},
			{ProviderReplicate, "bytedance/seedance-2.0"},
			{ProviderReplicate, "flux-2-pro"},
			{ProviderReplicate, "black-forest-labs/flux-2-pro"},
			{ProviderReplicate, "black-forest-labs/flux-2-pro:abc"},
			{ProviderReplicate, "some-owner/custom-model"},
		}
		for _, pair := range pairs {
			if err := CheckProviderModel(pair[0], pair[1]); err != nil {
				t.Errorf("CheckProviderModel(%q, %q) = %v, want nil", pair[0], pair[1], err)
			}
		}
	})
	t.Run("empty model or provider is skipped", func(t *testing.T) {
		if err := CheckProviderModel(ProviderXai, ""); err != nil {
			t.Errorf("empty model: %v", err)
		}
		if err := CheckProviderModel("", "gpt-image-2"); err != nil {
			t.Errorf("empty provider: %v", err)
		}
	})
}

func TestValidateIncompatiblePairBeforeFormat(t *testing.T) {
	// The UAT failure: xai + gpt-image-2 + mp4 used to fail with a
	// misleading output_format error after (or instead of) a paid request.
	r := Request{
		Provider:     ProviderXai,
		Token:        "tok",
		Model:        "gpt-image-2",
		Prompt:       "a fox",
		NumImages:    1,
		OutputFormat: "mp4",
	}
	err := r.Validate()
	if err == nil {
		t.Fatal("expected incompatibility error")
	}
	msg := err.Error()
	if strings.Contains(msg, "output_format") {
		t.Fatalf("got format error %q, want provider/model mismatch", msg)
	}
	if !strings.Contains(msg, ProviderXai) || !strings.Contains(msg, "gpt-image-2") || !strings.Contains(msg, DefaultXaiVideoModel) {
		t.Fatalf("error %q missing provider/model/supported", msg)
	}
}

type recordingProvider struct {
	calls int
}

func (p *recordingProvider) Name() string { return ProviderXai }

func (p *recordingProvider) Generate(context.Context, *Request) (*Result, error) {
	p.calls++
	return nil, nil
}

func TestGenerateIncompatiblePairMakesNoNetworkCall(t *testing.T) {
	rec := &recordingProvider{}
	c := &Client{Xai: rec}
	_, err := c.Generate(context.Background(), &Request{
		Provider:     ProviderXai,
		Token:        "tok",
		Model:        "gpt-image-2",
		Prompt:       "a fox",
		NumImages:    1,
		OutputFormat: "webp",
	})
	if err == nil {
		t.Fatal("expected incompatibility error")
	}
	if rec.calls != 0 {
		t.Fatalf("provider.Generate called %d times, want 0", rec.calls)
	}
	if !strings.Contains(err.Error(), "does not support model") {
		t.Fatalf("got %v", err)
	}
}

func TestApplyDefaultsXaiPicksTableDefault(t *testing.T) {
	r := Request{Provider: ProviderXai}
	r.applyDefaults()
	if r.Model != DefaultXaiVideoModel {
		t.Errorf("Model = %q, want %q", r.Model, DefaultXaiVideoModel)
	}
}
