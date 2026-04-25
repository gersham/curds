package tui

import (
	"context"
	"io"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
)

func newTestModel(_ *testing.T, settings SettingsValues, save SaveSettingsFn) model {
	d := Defaults{
		Provider:     "openai",
		AspectRatio:  "1:1",
		Quality:      "auto",
		NumImages:    1,
		OutputFormat: "webp",
	}
	gen := func(ctx context.Context, req GenerateRequest, sink io.Writer) GenerateResult {
		return GenerateResult{Paths: []string{"/tmp/x.webp"}}
	}
	return newModel(d, gen, settings, save)
}

// TestOpenSettingsRoutesInit is the regression test for the user report
// "enter doesn't work, can't get past Provider". The original bug was
// that openSettings called m.form.Init() but discarded its tea.Cmd, so
// the form never reached an interactive state. This test exercises the
// fix by checking that openSettings now returns a non-nil cmd alongside
// the model.
func TestOpenSettingsRoutesInit(t *testing.T) {
	initial := SettingsValues{
		Provider: "openai", AspectRatio: "1:1", Quality: "auto",
		NumberOfImages: 1, OutputFormat: "webp",
	}
	m := newTestModel(t, initial, func(SettingsValues) error { return nil })

	updated, cmd := m.openSettings()

	if updated.phase != phaseSettings {
		t.Fatalf("phase: got %d want %d", updated.phase, phaseSettings)
	}
	if updated.form == nil {
		t.Fatal("form should be created")
	}
	if updated.formValues == nil {
		t.Fatal("formValues should be created")
	}
	if cmd == nil {
		t.Fatal("openSettings must return form.Init() cmd, not discard it")
	}
}

// TestSettingsCtrlSEntersSettings verifies the ctrl+s key handler in
// phasePrompt correctly transitions into phaseSettings and returns a
// non-nil command (which will include the form's Init).
func TestSettingsCtrlSEntersSettings(t *testing.T) {
	initial := SettingsValues{
		Provider: "openai", AspectRatio: "1:1", Quality: "auto",
		NumberOfImages: 1, OutputFormat: "webp",
	}
	m := newTestModel(t, initial, func(SettingsValues) error { return nil })

	out, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlS})
	mm, ok := out.(model)
	if !ok {
		t.Fatalf("expected model, got %T", out)
	}
	if mm.phase != phaseSettings {
		t.Fatalf("ctrl+s did not enter settings: phase=%d", mm.phase)
	}
	if cmd == nil {
		t.Fatal("ctrl+s must return a cmd (ClearScreen + form.Init)")
	}
}

// TestSettingsCtrlCAborts verifies ctrl+c in phaseSettings returns to
// phasePrompt without invoking the save callback.
func TestSettingsCtrlCAborts(t *testing.T) {
	saveCh := make(chan SettingsValues, 1)
	save := func(v SettingsValues) error {
		saveCh <- v
		return nil
	}

	initial := SettingsValues{
		Provider: "openai", AspectRatio: "1:1", Quality: "auto",
		NumberOfImages: 1, OutputFormat: "webp",
	}
	m := newTestModel(t, initial, save)

	// Enter settings.
	mAny, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlS})
	m = mAny.(model)

	// ctrl+c while in settings.
	mAny, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = mAny.(model)

	if m.phase != phasePrompt {
		t.Fatalf("ctrl+c did not return to prompt: phase=%d", m.phase)
	}
	if m.form != nil {
		t.Fatal("form should be cleared after abort")
	}
	select {
	case <-saveCh:
		t.Fatal("save must not fire on ctrl+c")
	default:
	}
}

// TestApplySettingsParsesAndPersists feeds raw form values into the
// applySettings helper and confirms parsing + the save callback fire
// with the expected SettingsValues.
func TestApplySettingsParsesAndPersists(t *testing.T) {
	got := make(chan SettingsValues, 1)
	save := func(v SettingsValues) error {
		got <- v
		return nil
	}

	initial := SettingsValues{
		Provider: "openai", AspectRatio: "1:1", Quality: "auto",
		NumberOfImages: 1, OutputFormat: "webp", OutputCompression: 90,
	}
	m := newTestModel(t, initial, save)
	m.formValues = &settingsFormValues{
		Provider:          "replicate",
		OutputDirectory:   "/tmp/imgs",
		OutputFormat:      "png",
		OutputCompression: "75",
		Quality:           "high",
		AspectRatio:       "16:9",
		Background:        "opaque",
		Moderation:        "low",
		NumberOfImages:    "4",
		OpenAIToken:       "sk-x",
		ReplicateToken:    "r8_x",
		Confirmed:         true,
	}

	if err := m.applySettings(); err != nil {
		t.Fatalf("applySettings: %v", err)
	}

	v := <-got
	if v.Provider != "replicate" {
		t.Errorf("Provider: %q", v.Provider)
	}
	if v.AspectRatio != "16:9" {
		t.Errorf("AspectRatio: %q", v.AspectRatio)
	}
	if v.NumberOfImages != 4 {
		t.Errorf("NumberOfImages: %d", v.NumberOfImages)
	}
	if v.OutputCompression != 75 {
		t.Errorf("OutputCompression: %d", v.OutputCompression)
	}
	if v.OpenAIToken != "sk-x" || v.ReplicateToken != "r8_x" {
		t.Errorf("tokens not propagated: %+v", v)
	}

	// Defaults on the model should also be refreshed.
	if m.defaults.AspectRatio != "16:9" {
		t.Errorf("defaults.AspectRatio not refreshed: %q", m.defaults.AspectRatio)
	}
	if m.defaults.NumImages != 4 {
		t.Errorf("defaults.NumImages not refreshed: %d", m.defaults.NumImages)
	}
}

// Compile-time assertion that the model still satisfies tea.Model.
var _ tea.Model = model{}

var _ = huh.NewForm // ensure the import isn't dropped during refactors
