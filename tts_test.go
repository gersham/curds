package curds

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsTTSModel(t *testing.T) {
	cases := map[string]bool{
		"minimax/speech-2.8-hd":           true,
		"minimax/speech-2.8-hd:abc123":    true,
		"google/gemini-3.1-flash-tts":     true,
		"Google/Gemini-3.1-Flash-TTS ":    true,
		"elevenlabs/v3":                   true,
		"ElevenLabs/V3 ":                  true,
		"gpt-4o-mini-tts":                 true,
		"tts-1-hd":                        true,
		"elevenlabs/music":                false,
		"stability-ai/stable-audio-2.5":   false,
		"minimax/speech-2.8-hd-evil":      false,
		"elevenlabs/v3-pro":               false,
		"google/gemini-3.1-flash-tts-old": false,
		"gpt-4o-mini":                     false,
		"":                                false,
	}
	for in, want := range cases {
		if got := IsTTSModel(in); got != want {
			t.Errorf("IsTTSModel(%q) = %v want %v", in, got, want)
		}
	}
}

// TTS models are prompt-driven: they must not be exempted from the prompt
// requirement.
func TestTTSModelsAreNotPromptless(t *testing.T) {
	for _, model := range []string{TTSGeminiModel, TTSSpeechModel, TTSElevenLabsModel, TTSOpenAIModel, TTS1HDModel} {
		if IsPromptlessModel(model) {
			t.Errorf("IsPromptlessModel(%q) = true; TTS models need text", model)
		}
		if !IsAudioModel(model) {
			t.Errorf("IsAudioModel(%q) = false; TTS models emit Result.Audios", model)
		}
	}
}

func TestRequestValidateTTS(t *testing.T) {
	base := func(provider, model string) Request {
		return Request{
			Provider:     provider,
			Token:        "tok",
			Model:        model,
			Prompt:       "The rain had not stopped for a week.",
			NumImages:    1,
			OutputFormat: "mp3",
		}
	}
	cases := []struct {
		name       string
		provider   string
		model      string
		mut        func(r *Request)
		wantErrSub string
	}{
		{"gemini defaults", ProviderReplicate, TTSGeminiModel, func(r *Request) {}, ""},
		{"gemini voice and instructions", ProviderReplicate, TTSGeminiModel, func(r *Request) {
			r.Voice = "Puck"
			r.Instructions = "warm and slow, British accent"
		}, ""},
		{"gemini bad voice", ProviderReplicate, TTSGeminiModel, func(r *Request) { r.Voice = "Rachel" }, "voice must be one of"},
		{"gemini rejects speed", ProviderReplicate, TTSGeminiModel, func(r *Request) { r.Speed = 1.2 }, "-speed is not supported by -model tts"},
		{"gemini rejects emotion", ProviderReplicate, TTSGeminiModel, func(r *Request) { r.Emotion = "calm" }, "-emotion is only supported by -model tts-minimax"},
		{"gemini rejects pitch", ProviderReplicate, TTSGeminiModel, func(r *Request) { r.Pitch = 1 }, "-pitch is only supported by -model tts-minimax"},
		{"gemini rejects stability", ProviderReplicate, TTSGeminiModel, func(r *Request) { r.Stability = float64Ptr(0.5) }, "-stability and -style are only supported"},
		{"gemini needs replicate", ProviderOpenAI, TTSGeminiModel, func(r *Request) {}, "does not support model"},
		{"gemini text byte cap", ProviderReplicate, TTSGeminiModel, func(r *Request) {
			r.Prompt = strings.Repeat("a", MaxGeminiTTSBytes+1)
		}, "at most 4000"},
		{"gemini accepts exactly the byte cap", ProviderReplicate, TTSGeminiModel, func(r *Request) {
			r.Prompt = strings.Repeat("a", MaxGeminiTTSBytes)
		}, ""},
		{"gemini instructions byte cap", ProviderReplicate, TTSGeminiModel, func(r *Request) {
			r.Instructions = strings.Repeat("a", MaxGeminiTTSBytes+1)
		}, "style prompt accepts at most 4000"},
		{"gemini counts bytes not runes", ProviderReplicate, TTSGeminiModel, func(r *Request) {
			// 3000 two-byte runes are 6000 bytes: over the cap even though
			// the rune count is below it.
			r.Prompt = strings.Repeat("é", 3000)
		}, "at most 4000"},
		{"speech defaults", ProviderReplicate, TTSSpeechModel, func(r *Request) {}, ""},
		{"speech knobs", ProviderReplicate, TTSSpeechModel, func(r *Request) {
			r.Voice = "English_Deep-VoicedGentleman"
			r.Emotion = "calm"
			r.Speed = 1.25
			r.Pitch = -3
		}, ""},
		{"speech wav", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.OutputFormat = "wav" }, ""},
		{"speech needs replicate", ProviderOpenAI, TTSSpeechModel, func(r *Request) {}, "does not support model"},
		{"speech fast", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Speed = 2.5 }, "0.5-2"},
		{"speech pitch out of range", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Pitch = 13 }, "-12..12"},
		{"speech rejects instructions", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Instructions = "dry" }, "-instructions is only supported by -model tts (Gemini 3.1 Flash TTS) and -model tts-openai"},
		{"speech rejects stability", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Stability = float64Ptr(0.5) }, "-stability and -style are only supported"},
		{"speech rejects webp", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.OutputFormat = "webp" }, "mp3 or wav"},
		{"speech rejects size", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Size = "1024x1024" }, "no pixel size"},
		{"speech rejects input image", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.InputImages = []string{"a.png"} }, "no input media"},
		{"speech rejects two files", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.NumImages = 2 }, "exactly one file"},
		{"speech rejects lyrics", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Lyrics = "la la" }, "-lyrics is only supported"},
		{"speech rejects duration", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Duration = 10 }, "no -duration"},
		{"speech rejects instrumental", ProviderReplicate, TTSSpeechModel, func(r *Request) {
			instrumental := true
			r.Instrumental = &instrumental
		}, "-instrumental is only supported"},
		{"speech slow", ProviderReplicate, TTSSpeechModel, func(r *Request) { r.Speed = 0.4 }, "0.5-2"},
		{"speech text cap", ProviderReplicate, TTSSpeechModel, func(r *Request) {
			r.Prompt = strings.Repeat("a", MaxTTSTextChars+1)
		}, "at most 10000"},
		{"speech accepts exactly the cap", ProviderReplicate, TTSSpeechModel, func(r *Request) {
			r.Prompt = strings.Repeat("a", MaxTTSTextChars)
		}, ""},

		{"elevenlabs defaults", ProviderReplicate, TTSElevenLabsModel, func(r *Request) {}, ""},
		{"elevenlabs knobs", ProviderReplicate, TTSElevenLabsModel, func(r *Request) {
			r.Voice = "Rachel"
			r.Speed = 1.1
			r.Stability = float64Ptr(0)
			r.Style = float64Ptr(0.8)
		}, ""},
		{"elevenlabs bad voice", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Voice = "Bob" }, "voice must be one of"},
		{"elevenlabs speed range", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Speed = 1.5 }, "0.7-1.2"},
		{"elevenlabs stability range", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Stability = float64Ptr(1.5) }, "stability must be 0-1"},
		{"elevenlabs style range", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Style = float64Ptr(-0.5) }, "style must be 0-1"},
		{"elevenlabs rejects emotion", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Emotion = "happy" }, "-emotion is only supported"},
		{"elevenlabs rejects pitch", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Pitch = 2 }, "-pitch is only supported"},
		{"elevenlabs rejects instructions", ProviderReplicate, TTSElevenLabsModel, func(r *Request) { r.Instructions = "dry" }, "-instructions is only supported by -model tts (Gemini 3.1 Flash TTS) and -model tts-openai"},

		{"openai defaults", ProviderOpenAI, TTSOpenAIModel, func(r *Request) {}, ""},
		{"openai instructions", ProviderOpenAI, TTSOpenAIModel, func(r *Request) {
			r.Voice = "sage"
			r.Speed = 1.1
			r.Instructions = "crisp British RP, dry"
		}, ""},
		{"openai bad voice", ProviderOpenAI, TTSOpenAIModel, func(r *Request) { r.Voice = "Bob" }, "voice must be one of"},
		{"openai speed range", ProviderOpenAI, TTSOpenAIModel, func(r *Request) { r.Speed = 5 }, "0.25-4"},
		{"openai rejects emotion", ProviderOpenAI, TTSOpenAIModel, func(r *Request) { r.Emotion = "calm" }, "-emotion is only supported by -model tts-minimax"},
		{"openai rejects pitch", ProviderOpenAI, TTSOpenAIModel, func(r *Request) { r.Pitch = 1 }, "-pitch is only supported by -model tts-minimax"},
		{"openai rejects stability", ProviderOpenAI, TTSOpenAIModel, func(r *Request) { r.Style = float64Ptr(0.1) }, "-stability and -style are only supported"},
		{"openai needs openai", ProviderReplicate, TTSOpenAIModel, func(r *Request) {}, "does not support model"},
		{"openai text cap", ProviderOpenAI, TTSOpenAIModel, func(r *Request) {
			r.Prompt = strings.Repeat("a", MaxOpenAITTSChars+1)
		}, "at most 4096"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := base(tc.provider, tc.model)
			tc.mut(&r)
			err := r.Validate()
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

// TTS models need text like any prompt-driven model; there is no lyrics escape.
func TestRequestValidateTTSNeedsText(t *testing.T) {
	for _, model := range []string{TTSGeminiModel, TTSSpeechModel, TTSElevenLabsModel, TTSOpenAIModel} {
		provider := ProviderReplicate
		if IsOpenAITTSModel(model) {
			provider = ProviderOpenAI
		}
		r := Request{Provider: provider, Token: "tok", Model: model, NumImages: 1, OutputFormat: "mp3"}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "prompt is required") {
			t.Errorf("%s: got %v, want prompt is required", model, err)
		}
	}
}

func TestTTSModelDefaults(t *testing.T) {
	cases := map[string]string{
		TTSGeminiModel:     DefaultTTSGeminiVoice,
		TTSSpeechModel:     DefaultTTSVoice,
		TTSElevenLabsModel: DefaultTTSElevenLabsVoice,
		TTSOpenAIModel:     DefaultTTSOpenAIVoice,
		TTS1HDModel:        DefaultTTSOpenAIVoice,
	}
	for model, wantVoice := range cases {
		r := Request{Model: model}
		r.applyDefaults()
		if r.Voice != wantVoice {
			t.Errorf("%s voice: got %q want %q", model, r.Voice, wantVoice)
		}
		if r.OutputFormat != "mp3" {
			t.Errorf("%s OutputFormat: got %q want mp3", model, r.OutputFormat)
		}
		if r.AspectRatio != "" {
			t.Errorf("%s takes no aspect ratio, got %q", model, r.AspectRatio)
		}
	}
	// An explicit voice survives applyDefaults.
	r := Request{Model: TTSSpeechModel, Voice: "English_Deep-VoicedGentleman"}
	r.applyDefaults()
	if r.Voice != "English_Deep-VoicedGentleman" {
		t.Errorf("explicit voice overwritten: %q", r.Voice)
	}
}

func float64Ptr(v float64) *float64 { return &v }

func TestReplicateProviderSpeechHappyPath(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.mp3")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSSpeechModel,
		Prompt:       "Chapter one. The rain had not stopped.",
		Voice:        "English_Deep-VoicedGentleman",
		Emotion:      "calm",
		Speed:        1.25,
		Pitch:        -3,
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Audios) != 1 || res.Audios[0].Format != "mp3" {
		t.Fatalf("audios: %#v", res.Audios)
	}
	got := (*bodies)[0]
	if got["text"] != "Chapter one. The rain had not stopped." {
		t.Errorf("text: %#v", got["text"])
	}
	if got["voice_id"] != "English_Deep-VoicedGentleman" {
		t.Errorf("voice_id: %#v", got["voice_id"])
	}
	if got["emotion"] != "calm" || got["speed"] != 1.25 || got["pitch"] != float64(-3) {
		t.Errorf("delivery knobs: %#v", got)
	}
	if got["audio_format"] != "mp3" || got["sample_rate"] != float64(44100) || got["bitrate"] != float64(256000) {
		t.Errorf("output knobs: %#v", got)
	}
	if got["english_normalization"] != true {
		t.Errorf("english_normalization must be on: %#v", got["english_normalization"])
	}
	for _, forbidden := range []string{"prompt", "voice", "music_length_ms", "stability", "instructions"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("speech must not send %q: %#v", forbidden, got)
		}
	}
}

// Gemini 3.1 Flash TTS names its fields differently from the other speech
// models: the text goes in `text`, the voice in `voice`, and -instructions
// becomes the style prompt in `prompt`.
func TestReplicateProviderGeminiTTSHappyPath(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.wav")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSGeminiModel,
		Prompt:       "The results, I'm afraid, are conclusive.",
		Voice:        "Kore",
		Instructions: "warm and slow, British accent",
		NumImages:    1,
		OutputFormat: "wav",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Audios) != 1 || res.Audios[0].Format != "wav" {
		t.Fatalf("audios: %#v", res.Audios)
	}
	got := (*bodies)[0]
	if got["text"] != "The results, I'm afraid, are conclusive." {
		t.Errorf("text: %#v", got["text"])
	}
	if got["voice"] != "Kore" {
		t.Errorf("voice: %#v", got["voice"])
	}
	if got["prompt"] != "warm and slow, British accent" {
		t.Errorf("instructions must become the style prompt: %#v", got["prompt"])
	}
	// The model has no speed, emotion, pitch, stability, or container field.
	for _, forbidden := range []string{"speed", "emotion", "pitch", "stability", "style",
		"voice_id", "audio_format", "output_format", "sample_rate", "bitrate"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("gemini TTS must not send %q: %#v", forbidden, got)
		}
	}
}

// With no -instructions the style prompt is omitted so the model's own
// delivery default applies.
func TestReplicateProviderGeminiTTSOmitsUnsetPrompt(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.wav")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	if _, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSGeminiModel,
		Prompt:       "Just the text.",
		Voice:        DefaultTTSGeminiVoice,
		NumImages:    1,
		OutputFormat: "mp3",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := (*bodies)[0]
	if _, ok := got["prompt"]; ok {
		t.Errorf("unset style prompt must be omitted: %#v", got)
	}
	if got["voice"] != DefaultTTSGeminiVoice {
		t.Errorf("voice: %#v", got["voice"])
	}
}

// WAV output has no bitrate knob and must not carry one.
func TestReplicateProviderSpeechWAVFormat(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.wav")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	if _, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSSpeechModel,
		Prompt:       "Narration.",
		Voice:        DefaultTTSVoice,
		NumImages:    1,
		OutputFormat: "wav",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := (*bodies)[0]
	if got["audio_format"] != "wav" {
		t.Errorf("audio_format: %#v", got["audio_format"])
	}
	if _, ok := got["bitrate"]; ok {
		t.Errorf("wav must not send bitrate: %#v", got)
	}
	if got["sample_rate"] != float64(44100) {
		t.Errorf("sample_rate: %#v", got["sample_rate"])
	}
}

func TestReplicateProviderElevenLabsTTSHappyPath(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/line.mp3")

	stability := 0.0
	style := 0.8
	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSElevenLabsModel,
		Prompt:       "Oh, brilliant. Another Monday.",
		Voice:        "Rachel",
		Speed:        1.1,
		Stability:    &stability,
		Style:        &style,
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Audios) != 1 || res.Audios[0].Format != "mp3" {
		t.Fatalf("audios: %#v", res.Audios)
	}
	got := (*bodies)[0]
	if got["prompt"] != "Oh, brilliant. Another Monday." {
		t.Errorf("prompt (the text) must pass through: %#v", got["prompt"])
	}
	if got["voice"] != "Rachel" {
		t.Errorf("voice: %#v", got["voice"])
	}
	if got["speed"] != 1.1 || got["stability"] != float64(0) || got["style"] != 0.8 {
		t.Errorf("knobs: %#v", got)
	}
	// The model picks its own container; curds does not send a format.
	for _, forbidden := range []string{"text", "voice_id", "audio_format", "sample_rate", "bitrate"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("elevenlabs TTS must not send %q: %#v", forbidden, got)
		}
	}
}

// With no stability/style the fields are omitted so the model's own defaults
// apply.
func TestReplicateProviderElevenLabsTTSOmitsUnsetKnobs(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/line.mp3")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	if _, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        TTSElevenLabsModel,
		Prompt:       "Just the text.",
		Voice:        DefaultTTSElevenLabsVoice,
		NumImages:    1,
		OutputFormat: "mp3",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := (*bodies)[0]
	for _, forbidden := range []string{"speed", "stability", "style"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("unset %q must be omitted: %#v", forbidden, got)
		}
	}
}

// openAISpeechServer records the speech request and returns raw audio bytes.
func openAISpeechServer(t *testing.T, body []byte, status int) (*httptest.Server, *[]map[string]any, *[]string, *[]string) {
	t.Helper()
	bodies := &[]map[string]any{}
	auths := &[]string{}
	paths := &[]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*paths = append(*paths, r.URL.Path)
		*auths = append(*auths, r.Header.Get("Authorization"))
		raw, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(raw, &b)
		*bodies = append(*bodies, b)
		w.Header().Set("Content-Type", "audio/mpeg")
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, bodies, auths, paths
}

func TestOpenAISpeechHappyPath(t *testing.T) {
	audio := []byte("ID3-not-really-mp3")
	srv, bodies, auths, paths := openAISpeechServer(t, audio, http.StatusOK)

	p := &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderOpenAI,
		Token:        "otok",
		Model:        TTSOpenAIModel,
		Prompt:       "The results, I'm afraid, are conclusive.",
		Voice:        "sage",
		Speed:        1.1,
		Instructions: "crisp British RP, dry",
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if (*paths)[0] != "/audio/speech" {
		t.Errorf("endpoint path: %q", (*paths)[0])
	}
	if (*auths)[0] != "Bearer otok" {
		t.Errorf("Authorization: %q", (*auths)[0])
	}
	got := (*bodies)[0]
	if got["model"] != TTSOpenAIModel || got["input"] != "The results, I'm afraid, are conclusive." ||
		got["voice"] != "sage" || got["response_format"] != "mp3" {
		t.Errorf("speech body: %#v", got)
	}
	if got["speed"] != 1.1 || got["instructions"] != "crisp British RP, dry" {
		t.Errorf("speech knobs: %#v", got)
	}
	if len(res.Audios) != 1 {
		t.Fatalf("audios: %#v", res.Audios)
	}
	if string(res.Audios[0].Bytes) != string(audio) {
		t.Errorf("audio bytes must be the raw response body: %q", res.Audios[0].Bytes)
	}
	if res.Audios[0].Format != "mp3" {
		t.Errorf("format: %q", res.Audios[0].Format)
	}
}

// tts-1-hd omits instructions, and there is no local ffmpeg between curds and
// the bytes: the response body is the file.
func TestOpenAISpeechNoInstructionsAndWAV(t *testing.T) {
	srv, bodies, _, _ := openAISpeechServer(t, []byte("RIFFxxxxWAVE"), http.StatusOK)

	p := &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderOpenAI,
		Token:        "otok",
		Model:        TTS1HDModel,
		Prompt:       "Narration.",
		Voice:        DefaultTTSOpenAIVoice,
		NumImages:    1,
		OutputFormat: "wav",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := (*bodies)[0]
	if got["response_format"] != "wav" {
		t.Errorf("response_format: %#v", got["response_format"])
	}
	for _, forbidden := range []string{"instructions", "speed"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("unset %q must be omitted: %#v", forbidden, got)
		}
	}
	if res.Audios[0].Format != "wav" {
		t.Errorf("format: %q", res.Audios[0].Format)
	}
}

// The speech endpoint reports errors as JSON; curds surfaces the message.
func TestOpenAISpeechAPIError(t *testing.T) {
	errBody := []byte(`{"error":{"message":"Invalid voice: nope","type":"invalid_request_error","code":"invalid_voice"}}`)
	srv, _, _, _ := openAISpeechServer(t, errBody, http.StatusBadRequest)

	p := &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	_, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderOpenAI,
		Token:        "otok",
		Model:        TTSOpenAIModel,
		Prompt:       "hi",
		Voice:        "sage",
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err == nil || !strings.Contains(err.Error(), "Invalid voice: nope") {
		t.Fatalf("got %v, want the upstream message", err)
	}
}

// An empty 200 body is a failure, not a zero-byte audio file.
func TestOpenAISpeechEmptyBody(t *testing.T) {
	srv, _, _, _ := openAISpeechServer(t, nil, http.StatusOK)

	p := &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	_, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderOpenAI,
		Token:        "otok",
		Model:        TTSOpenAIModel,
		Prompt:       "hi",
		Voice:        "sage",
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err == nil || !strings.Contains(err.Error(), "no audio") {
		t.Fatalf("got %v, want an empty-body error", err)
	}
}

// The voice enums are the documented upstream lists; a mismatch means curds
// rejects a valid voice or accepts an invalid one.
func TestTTSVoiceEnums(t *testing.T) {
	for _, v := range []string{"Rachel", "Drew", "Grimblewood"} {
		if !ElevenLabsTTSVoices[v] {
			t.Errorf("ElevenLabsTTSVoices missing documented voice %q", v)
		}
	}
	if len(ElevenLabsTTSVoices) != 26 {
		t.Errorf("ElevenLabsTTSVoices has %d entries, want 26", len(ElevenLabsTTSVoices))
	}
	// The Gemini enum is the 30-voice list from the model's schema.
	for _, v := range []string{"Kore", "Puck", "Zephyr", "Zubenelgenubi"} {
		if !GeminiTTSVoices[v] {
			t.Errorf("GeminiTTSVoices missing documented voice %q", v)
		}
	}
	if len(GeminiTTSVoices) != 30 {
		t.Errorf("GeminiTTSVoices has %d entries, want 30", len(GeminiTTSVoices))
	}
	for _, v := range []string{"alloy", "sage", "cedar", "marin"} {
		if !OpenAITTSVoices[v] {
			t.Errorf("OpenAITTSVoices missing documented voice %q", v)
		}
	}
	for _, e := range []string{"auto", "happy", "neutral", "fluent"} {
		if !MinimaxTTSEmotions[e] {
			t.Errorf("MinimaxTTSEmotions missing documented emotion %q", e)
		}
	}
}
