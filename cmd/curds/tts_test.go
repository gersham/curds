package main

import (
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gersham/curds"
)

// withClient points the generation path at a stub client for the duration of a
// test, the same seam newRunProvider gives `curds run`.
func withClient(t *testing.T, c *curds.Client) {
	t.Helper()
	old := newClient
	newClient = func() *curds.Client { return c }
	t.Cleanup(func() { newClient = old })
}

// ttsTestEnv points the config at a throwaway file and clears every token env
// var so provider detection is deterministic.
func ttsTestEnv(t *testing.T) {
	t.Helper()
	t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))
	for _, v := range []string{"OPENAI_API_KEY", "REPLICATE_API_TOKEN", "XAI_API_KEY"} {
		t.Setenv(v, "")
	}
}

// runRealMain drives the flag-parsing entry point with fresh args and a fresh
// flag set (the global one cannot be re-registered).
func runRealMain(t *testing.T, args []string) (string, error) {
	t.Helper()
	oldArgs, oldCmd := os.Args, flag.CommandLine
	flag.CommandLine = flag.NewFlagSet("curds", flag.ContinueOnError)
	os.Args = append([]string{"curds"}, args...)
	t.Cleanup(func() {
		os.Args, flag.CommandLine = oldArgs, oldCmd
	})
	logger := newLogger(io.Discard)
	var err error
	printed := captureStdout(t, func() { err = realMain(logger, time.Now()) })
	return printed, err
}

// replicateAudioStub stands in for Replicate: it records the prediction input
// and answers with one audio URL served by a sibling assets server.
func replicateAudioStub(t *testing.T, ext string) (*curds.ReplicateProvider, *[]map[string]any) {
	t.Helper()
	assets := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("audio-bytes-" + ext))
	}))
	t.Cleanup(assets.Close)

	inputs := &[]map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(body, &b)
		if in, ok := b["input"].(map[string]any); ok {
			*inputs = append(*inputs, in)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "tts-1", "status": "succeeded", "output": assets.URL + "/line." + ext,
		})
	}))
	t.Cleanup(srv.Close)
	return &curds.ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}, inputs
}

// openAISpeechStub stands in for POST /v1/audio/speech: it records the JSON
// body and Authorization header, and returns raw audio bytes.
func openAISpeechStub(t *testing.T) (*curds.OpenAIProvider, *[]map[string]any, *[]string) {
	t.Helper()
	bodies := &[]map[string]any{}
	auths := &[]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(raw, &b)
		*bodies = append(*bodies, b)
		*auths = append(*auths, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("openai-audio-bytes"))
	}))
	t.Cleanup(srv.Close)
	return &curds.OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL}, bodies, auths
}

func TestValidateTTSFlags(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		mut        func(o *cliOptions)
		wantErrSub string
	}{
		{"tts defaults", curds.TTSSpeechModel, func(o *cliOptions) {}, ""},
		{"tts emotion speed pitch", curds.TTSSpeechModel, func(o *cliOptions) {
			o.emotion = "calm"
			o.pitch = -3
			o.speed = 1.5
		}, ""},
		{"tts voice is free-form", curds.TTSSpeechModel, func(o *cliOptions) { o.voice = "MyClonedVoice" }, ""},
		{"tts bad emotion", curds.TTSSpeechModel, func(o *cliOptions) { o.emotion = "sleepy" }, "-emotion must be one of"},
		{"tts bad speed", curds.TTSSpeechModel, func(o *cliOptions) { o.speed = 0.4 }, "-speed must be 0.5-2"},
		{"tts bad pitch", curds.TTSSpeechModel, func(o *cliOptions) { o.pitch = 13 }, "-pitch must be -12..12"},
		{"tts rejects instructions", curds.TTSSpeechModel, func(o *cliOptions) { o.instructions = "dry" }, "-instructions is only supported by -model tts-openai"},
		{"tts text cap", curds.TTSSpeechModel, func(o *cliOptions) {
			o.prompt = strings.Repeat("a", curds.MaxTTSTextChars+1)
		}, "at most 10000"},
		{"tts rejects duration", curds.TTSSpeechModel, func(o *cliOptions) { o.duration = 10 }, "-duration is only supported"},
		{"tts rejects lyrics", curds.TTSSpeechModel, func(o *cliOptions) { o.lyrics = "la la" }, "-lyrics is only supported"},

		{"elevenlabs defaults", curds.TTSElevenLabsModel, func(o *cliOptions) {}, ""},
		{"elevenlabs named voice", curds.TTSElevenLabsModel, func(o *cliOptions) { o.voice = "Rachel" }, ""},
		{"elevenlabs bad voice", curds.TTSElevenLabsModel, func(o *cliOptions) { o.voice = "Bob" }, "is not a valid -model tts-elevenlabs voice"},
		{"elevenlabs bad speed", curds.TTSElevenLabsModel, func(o *cliOptions) { o.speed = 1.5 }, "-speed must be 0.7-1.2"},
		{"elevenlabs rejects emotion", curds.TTSElevenLabsModel, func(o *cliOptions) { o.emotion = "happy" }, "-emotion is only supported by -model tts"},

		{"openai defaults", curds.TTSOpenAIModel, func(o *cliOptions) {}, ""},
		{"openai voice", curds.TTSOpenAIModel, func(o *cliOptions) { o.voice = "sage" }, ""},
		{"openai bad voice", curds.TTSOpenAIModel, func(o *cliOptions) { o.voice = "Bob" }, "is not a valid OpenAI speech voice"},
		{"openai instructions", curds.TTSOpenAIModel, func(o *cliOptions) { o.instructions = "crisp British RP, dry" }, ""},
		{"tts-1-hd rejects instructions", curds.TTS1HDModel, func(o *cliOptions) { o.instructions = "dry" }, "only supported by gpt-4o-mini-tts"},
		{"openai bad speed", curds.TTSOpenAIModel, func(o *cliOptions) { o.speed = 5 }, "-speed must be 0.25-4"},
		{"openai text cap", curds.TTSOpenAIModel, func(o *cliOptions) {
			o.prompt = strings.Repeat("a", curds.MaxOpenAITTSChars+1)
		}, "at most 4096"},

		{"music rejects voice", curds.MusicModel, func(o *cliOptions) { o.voice = "Rachel" }, "need a text-to-speech model"},
		{"image rejects speed", "gpt-image-2.5", func(o *cliOptions) { o.speed = 1 }, "need a text-to-speech model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts := &cliOptions{prompt: "The rain had not stopped for a week.", stability: -1, style: -1}
			tc.mut(opts)
			err := validateAudioFlags(tc.model, opts)
			if tc.wantErrSub == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("got %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

// TTS flag problems must be usage errors (exit 2), decided before any network
// call or TUI prompt.
func TestTTSFlagsExitTwo(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"instructions on minimax", []string{"-no-tui", "-model", "tts", "-instructions", "x", "-prompt", "hi"}, "-instructions is only supported by -model tts-openai"},
		{"bad elevenlabs voice", []string{"-no-tui", "-model", "tts-elevenlabs", "-voice", "Bob", "-prompt", "hi"}, "is not a valid -model tts-elevenlabs voice"},
		{"stability on minimax", []string{"-no-tui", "-model", "tts", "-stability", "0.5", "-prompt", "hi"}, "-stability and -style are only supported"},
		{"style on minimax", []string{"-no-tui", "-model", "tts", "-style", "0", "-prompt", "hi"}, "-stability and -style are only supported"},
		{"emotion on openai", []string{"-no-tui", "-model", "tts-openai", "-emotion", "calm", "-prompt", "hi"}, "-emotion is only supported by -model tts"},
		{"voice on an image model", []string{"-no-tui", "-model", "gpt-image-2.5", "-voice", "sage", "-prompt", "hi"}, "need a text-to-speech model"},
		{"missing text", []string{"-no-tui", "-model", "tts"}, "prompt is required"},
		{"emotion outside the enum", []string{"-no-tui", "-model", "tts", "-emotion", "sleepy", "-prompt", "hi"}, "-emotion must be one of"},
		{"instructions on tts-1-hd", []string{"-no-tui", "-model", "tts-1-hd", "-instructions", "dry", "-prompt", "hi"}, "only supported by gpt-4o-mini-tts"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ttsTestEnv(t)
			_, err := runRealMain(t, tc.args)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want substring %q", err, tc.want)
			}
			var ue *usageError
			if !errors.As(err, &ue) {
				t.Errorf("TTS flag problems must exit 2, got %T", err)
			}
		})
	}
}

// The resolved model decides the provider and the input shape. Both providers
// are stubbed and both token env vars are primed, so a mis-route is caught.
func TestTTSProviderRouting(t *testing.T) {
	newStubs := func(t *testing.T, ext string) (*curds.Client, *[]map[string]any, *[]map[string]any, *[]string) {
		t.Helper()
		rep, repInputs := replicateAudioStub(t, ext)
		openai, bodies, auths := openAISpeechStub(t)
		return &curds.Client{Replicate: rep, OpenAI: openai}, repInputs, bodies, auths
	}

	t.Run("tts routes to replicate via the speech builder", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		t.Setenv("OPENAI_API_KEY", "otok-env")
		c, repInputs, bodies, _ := newStubs(t, "mp3")
		withClient(t, c)

		out := filepath.Join(t.TempDir(), "line.mp3")
		printed, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts", "-prompt", "Chapter one. The rain had not stopped.",
			"-voice", "English_Deep-VoicedGentleman", "-emotion", "calm",
			"-speed", "1.25", "-pitch", "-3", "-output", out,
		})
		if err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*bodies) != 0 {
			t.Errorf("replicate model must not hit the OpenAI speech endpoint")
		}
		if len(*repInputs) != 1 {
			t.Fatalf("replicate inputs: %d", len(*repInputs))
		}
		in := (*repInputs)[0]
		if in["text"] != "Chapter one. The rain had not stopped." || in["voice_id"] != "English_Deep-VoicedGentleman" {
			t.Errorf("speech input: %#v", in)
		}
		if in["emotion"] != "calm" || in["speed"] != 1.25 || in["pitch"] != float64(-3) {
			t.Errorf("delivery knobs: %#v", in)
		}
		if in["audio_format"] != "mp3" || in["sample_rate"] != float64(44100) || in["bitrate"] != float64(256000) {
			t.Errorf("output knobs: %#v", in)
		}
		if in["english_normalization"] != true {
			t.Errorf("english_normalization: %#v", in)
		}
		data, readErr := os.ReadFile(out)
		if readErr != nil || string(data) != "audio-bytes-mp3" {
			t.Errorf("saved file: %q err=%v", data, readErr)
		}
		if !strings.Contains(printed, out) {
			t.Errorf("stdout should carry the path, got %q", printed)
		}
	})

	t.Run("tts-elevenlabs routes to replicate with stability/style", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		t.Setenv("OPENAI_API_KEY", "otok-env")
		c, repInputs, bodies, _ := newStubs(t, "mp3")
		withClient(t, c)

		out := filepath.Join(t.TempDir(), "line.mp3")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts-elevenlabs", "-prompt", "[sarcastic] Oh, brilliant.",
			"-voice", "Rachel", "-stability", "0", "-style", "0.8", "-speed", "1.1", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*bodies) != 0 {
			t.Errorf("elevenlabs model must not hit the OpenAI speech endpoint")
		}
		in := (*repInputs)[0]
		if in["prompt"] != "[sarcastic] Oh, brilliant." || in["voice"] != "Rachel" {
			t.Errorf("elevenlabs input: %#v", in)
		}
		if in["stability"] != float64(0) || in["style"] != 0.8 || in["speed"] != 1.1 {
			t.Errorf("knobs (0 stability must survive): %#v", in)
		}
		if data, err := os.ReadFile(out); err != nil || string(data) != "audio-bytes-mp3" {
			t.Errorf("saved file: %q err=%v", data, err)
		}
	})

	t.Run("tts-openai routes to the openai speech endpoint", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("OPENAI_API_KEY", "otok-env")
		c, repInputs, bodies, auths := newStubs(t, "mp3")
		withClient(t, c)

		out := filepath.Join(t.TempDir(), "line.mp3")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts-openai", "-prompt", "The results are conclusive.",
			"-voice", "sage", "-instructions", "crisp British RP, dry", "-speed", "1.2", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*repInputs) != 0 {
			t.Errorf("openai model must not hit Replicate")
		}
		if len(*bodies) != 1 {
			t.Fatalf("openai bodies: %d", len(*bodies))
		}
		body := (*bodies)[0]
		if body["model"] != curds.TTSOpenAIModel || body["input"] != "The results are conclusive." {
			t.Errorf("speech body: %#v", body)
		}
		if body["voice"] != "sage" || body["response_format"] != "mp3" || body["speed"] != 1.2 {
			t.Errorf("speech knobs: %#v", body)
		}
		if body["instructions"] != "crisp British RP, dry" {
			t.Errorf("instructions must be sent for gpt-4o-mini-tts: %#v", body)
		}
		if (*auths)[0] != "Bearer otok-env" {
			t.Errorf("Authorization: %q", (*auths)[0])
		}
		if data, err := os.ReadFile(out); err != nil || string(data) != "openai-audio-bytes" {
			t.Errorf("saved file: %q err=%v", data, err)
		}
	})

	t.Run("tts-1-hd routes to openai without instructions", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("OPENAI_API_KEY", "otok-env")
		c, _, bodies, _ := newStubs(t, "mp3")
		withClient(t, c)

		out := filepath.Join(t.TempDir(), "line.mp3")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts-1-hd", "-prompt", "Narration.", "-voice", "alloy", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		body := (*bodies)[0]
		if body["model"] != curds.TTS1HDModel || body["voice"] != "alloy" {
			t.Errorf("speech body: %#v", body)
		}
		if _, ok := body["instructions"]; ok {
			t.Errorf("tts-1-hd takes no instructions: %#v", body)
		}
	})

	t.Run("a .wav output asks for wav", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		c, repInputs, _, _ := newStubs(t, "wav")
		withClient(t, c)

		out := filepath.Join(t.TempDir(), "line.wav")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts", "-prompt", "Narration.", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		in := (*repInputs)[0]
		if in["audio_format"] != "wav" {
			t.Errorf("-output .wav must ask for wav, got %#v", in["audio_format"])
		}
		if _, ok := in["bitrate"]; ok {
			t.Errorf("wav has no bitrate: %#v", in)
		}
		if in["voice_id"] != curds.DefaultTTSVoice {
			t.Errorf("default voice: %#v", in["voice_id"])
		}
		if data, err := os.ReadFile(out); err != nil || string(data) != "audio-bytes-wav" {
			t.Errorf("saved file: %q err=%v", data, err)
		}
	})

	t.Run("stdin text is read for TTS", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("OPENAI_API_KEY", "otok-env")
		c, _, bodies, _ := newStubs(t, "mp3")
		withClient(t, c)

		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.WriteString("Read from stdin.\n"); err != nil {
			t.Fatal(err)
		}
		_ = w.Close()
		oldStdin := os.Stdin
		os.Stdin = r
		t.Cleanup(func() { os.Stdin = oldStdin })

		out := filepath.Join(t.TempDir(), "line.mp3")
		if _, err := runRealMain(t, []string{"-no-tui", "-model", "tts-openai", "-output", out}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if got := (*bodies)[0]["input"]; got != "Read from stdin." {
			t.Errorf("stdin text must become the speech input, got %#v", got)
		}
	})
}

// The help is the contract users and agents read.
func TestHelpTextMentionsTTS(t *testing.T) {
	help := helpText()
	for _, want := range []string{
		"TEXT TO SPEECH",
		"Text to speech",
		"-model tts ",
		"-model tts-elevenlabs",
		"-model tts-openai",
		"-model tts-1-hd",
		"minimax/speech-2.8-hd",
		"elevenlabs/v3",
		"gpt-4o-mini-tts",
		"-voice NAME",
		"-emotion VALUE",
		"-speed N",
		"-pitch N",
		"-instructions TEXT",
		"-stability N",
		"-style N",
		"crisp British RP, dry",
		"curds run -schema minimax/speech-2.8-hd",
		"cat script.txt | curds -model tts",
	} {
		if !strings.Contains(help, want) {
			t.Errorf("help is missing %q", want)
		}
	}
}
