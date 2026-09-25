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

func TestIsAudioModel(t *testing.T) {
	cases := map[string]bool{
		"elevenlabs/music":                   true,
		"elevenlabs/music:abc123":            true,
		"minimax/music-2.6":                  true,
		"stability-ai/stable-audio-2.5":      true,
		"stability-ai/stable-audio-2.5:v1":   true,
		"ElevenLabs/Music ":                  true,
		"elevenlabs/music-remix":             false,
		"black-forest-labs/flux-2-pro":       false,
		"stability-ai/stable-audio-2.5-evil": false,
		"kwaivgi/kling-avatar-v2":            false,
		"":                                   false,
	}
	for in, want := range cases {
		if got := IsAudioModel(in); got != want {
			t.Errorf("IsAudioModel(%q) = %v want %v", in, got, want)
		}
	}
}

// Audio models need a prompt (or, for music-vocal, lyrics), so they must not
// be treated as promptless.
func TestAudioModelsAreNotPromptless(t *testing.T) {
	for _, model := range []string{MusicModel, MusicVocalModel, SFXModel} {
		if IsPromptlessModel(model) {
			t.Errorf("IsPromptlessModel(%q) = true; audio models need a prompt", model)
		}
	}
}

func TestRequestValidateAudio(t *testing.T) {
	base := func(model string) Request {
		return Request{
			Provider:     ProviderReplicate,
			Token:        "rtok",
			Model:        model,
			Prompt:       "a slow ambient pad, no percussion",
			NumImages:    1,
			OutputFormat: "mp3",
		}
	}
	cases := []struct {
		name       string
		model      string
		mut        func(r *Request)
		wantErrSub string
	}{
		{"music prompt and mp3", MusicModel, func(r *Request) {}, ""},
		{"music exact length", MusicModel, func(r *Request) { r.Duration = 45 }, ""},
		{"music wav", MusicModel, func(r *Request) { r.OutputFormat = "wav" }, ""},
		{"music duration below 5s", MusicModel, func(r *Request) { r.Duration = 4 }, "5-300"},
		{"music duration above 300s", MusicModel, func(r *Request) { r.Duration = 301 }, "5-300"},
		{"music rejects lyrics", MusicModel, func(r *Request) { r.Lyrics = "[Verse]\nhi" }, "-lyrics is only supported by -model music-vocal"},
		{"music rejects seed", MusicModel, func(r *Request) { r.Seed = 3 }, "no seed input"},
		{"music rejects webp", MusicModel, func(r *Request) { r.OutputFormat = "webp" }, "mp3 or wav"},
		{"music rejects size", MusicModel, func(r *Request) { r.Size = "1024x1024" }, "no pixel size"},
		{"music rejects mask", MusicModel, func(r *Request) { r.Mask = "m.png" }, "-mask"},
		{"music rejects input image", MusicModel, func(r *Request) { r.InputImages = []string{"a.png"} }, "no input media"},
		{"music rejects input video", MusicModel, func(r *Request) { r.InputVideo = "clip.mp4" }, "no input media"},
		{"music rejects two files", MusicModel, func(r *Request) { r.NumImages = 2 }, "exactly one file"},
		{"music rejects xai", MusicModel, func(r *Request) { r.Provider = ProviderXai }, "does not support model"},

		{"music-vocal prompt", MusicVocalModel, func(r *Request) {}, ""},
		{"music-vocal lyrics without prompt", MusicVocalModel, func(r *Request) {
			r.Prompt = ""
			r.Lyrics = "[Verse]\nunder the streetlight"
		}, ""},
		{"music-vocal instrumental with prompt", MusicVocalModel, func(r *Request) {
			instrumental := true
			r.Instrumental = &instrumental
		}, ""},
		{"music-vocal needs prompt or lyrics", MusicVocalModel, func(r *Request) { r.Prompt = "" }, "prompt is required"},
		{"music-vocal sub-second duration", MusicVocalModel, func(r *Request) { r.Duration = 0.5 }, "at least 1 second"},
		{"music-vocal accepts seed rejection", MusicVocalModel, func(r *Request) { r.Seed = 1 }, "no seed input"},

		{"sfx prompt", SFXModel, func(r *Request) {}, ""},
		{"sfx duration and seed", SFXModel, func(r *Request) {
			r.Seed = 7
		}, ""},
		{"sfx duration below 1s", SFXModel, func(r *Request) { r.Duration = 0.4 }, "1-190"},
		{"sfx duration above 190s", SFXModel, func(r *Request) { r.Duration = 191 }, "1-190"},
		{"sfx rejects lyrics", SFXModel, func(r *Request) { r.Lyrics = "la la" }, "-lyrics is only supported by -model music-vocal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := base(tc.model)
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

// A music-vocal request that carries lyrics but no prompt is the one audio
// case that satisfies the prompt requirement without a prompt.
func TestAudioPromptRequirement(t *testing.T) {
	r := Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        MusicVocalModel,
		Lyrics:       "[Chorus]\nhold the line",
		NumImages:    1,
		OutputFormat: "mp3",
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("lyrics alone must validate: %v", err)
	}
	r.Lyrics = ""
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "prompt is required") {
		t.Fatalf("want prompt requirement, got %v", err)
	}
	// Lyrics on a model that cannot sing stay a prompt error for the music
	// model too (its own validator names the flag, but the outer prompt check
	// runs first when there is no prompt).
	r.Model = MusicModel
	r.Lyrics = "la la"
	if err := r.Validate(); err == nil {
		t.Fatal("music + lyrics without a prompt must fail")
	}
}

// validateAudio keeps its own copy of the music-vocal prompt rule for callers
// that skip the outer prompt check.
func TestValidateAudioMusicVocalNeedsInput(t *testing.T) {
	r := Request{
		Provider:     ProviderReplicate,
		Model:        MusicVocalModel,
		NumImages:    1,
		OutputFormat: "mp3",
	}
	err := r.validateAudio()
	if err == nil || !strings.Contains(err.Error(), "needs -prompt") {
		t.Fatalf("got %v, want the music-vocal prompt/lyrics rule", err)
	}
}

func TestAudioModelDefaults(t *testing.T) {
	t.Run("music is instrumental mp3", func(t *testing.T) {
		r := Request{Model: MusicModel}
		r.applyDefaults()
		if r.OutputFormat != "mp3" {
			t.Errorf("OutputFormat: got %q", r.OutputFormat)
		}
		if r.AspectRatio != "" {
			t.Errorf("audio takes no aspect ratio, got %q", r.AspectRatio)
		}
		if r.Instrumental == nil || !*r.Instrumental {
			t.Errorf("music must default to instrumental: %#v", r.Instrumental)
		}
		if r.Duration != 0 {
			t.Errorf("music duration stays unset for the model's own default, got %g", r.Duration)
		}
	})
	t.Run("music-vocal is vocal", func(t *testing.T) {
		r := Request{Model: MusicVocalModel}
		r.applyDefaults()
		if r.Instrumental == nil || *r.Instrumental {
			t.Errorf("music-vocal must default to vocals: %#v", r.Instrumental)
		}
	})
	t.Run("lyrics force vocals", func(t *testing.T) {
		instrumental := true
		r := Request{Model: MusicVocalModel, Instrumental: &instrumental, Lyrics: "[Verse]\nhi"}
		r.applyDefaults()
		if *r.Instrumental {
			t.Error("lyrics must override -instrumental=true")
		}
	})
	t.Run("sfx gets the default length", func(t *testing.T) {
		r := Request{Model: SFXModel}
		r.applyDefaults()
		if r.Duration != DefaultSFXDuration {
			t.Errorf("sfx Duration: got %g want %d", r.Duration, DefaultSFXDuration)
		}
	})
	t.Run("explicit format survives", func(t *testing.T) {
		r := Request{Model: MusicModel, OutputFormat: "wav"}
		r.applyDefaults()
		if r.OutputFormat != "wav" {
			t.Errorf("OutputFormat: got %q", r.OutputFormat)
		}
	})
}

func TestAudioFormatHelpers(t *testing.T) {
	if got := elevenLabsMusicFormat("mp3"); got != "mp3_high_quality" {
		t.Errorf("mp3 mapping: %q", got)
	}
	if got := elevenLabsMusicFormat("wav"); got != "wav_cd_quality" {
		t.Errorf("wav mapping: %q", got)
	}
	cases := []struct {
		url, fallback, want string
	}{
		{"https://replicate.delivery/out.wav", "mp3", "wav"},
		{"https://replicate.delivery/out.mp3", "wav", "mp3"},
		{"https://replicate.delivery/out", "mp3", "mp3"},
		{"https://replicate.delivery/out.wav?token=1", "mp3", "wav"},
	}
	for _, tc := range cases {
		if got := audioFormatFromURL(tc.url, tc.fallback); got != tc.want {
			t.Errorf("audioFormatFromURL(%q, %q) = %q want %q", tc.url, tc.fallback, got, tc.want)
		}
	}
}

// audioServer answers every prediction with one audio URL and records inputs.
func audioServer(t *testing.T, output string) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	bodies := &[]map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var b map[string]any
		if err := json.Unmarshal(body, &b); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		input, _ := b["input"].(map[string]any)
		*bodies = append(*bodies, input)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "aud-1", "status": "succeeded", "output": output,
		})
	}))
	t.Cleanup(srv.Close)
	return srv, bodies
}

func TestReplicateProviderMusicHappyPath(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.mp3")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        MusicModel,
		Prompt:       "tense minimal synth score",
		Duration:     45,
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Audios) != 1 {
		t.Fatalf("audios: %#v", res.Audios)
	}
	if res.Audios[0].Format != "mp3" {
		t.Errorf("format: %q", res.Audios[0].Format)
	}
	got := (*bodies)[0]
	if got["prompt"] != "tense minimal synth score" {
		t.Errorf("prompt: %#v", got["prompt"])
	}
	if got["music_length_ms"] != float64(45000) {
		t.Errorf("music_length_ms: %#v", got["music_length_ms"])
	}
	if got["force_instrumental"] != true {
		t.Errorf("force_instrumental: %#v", got["force_instrumental"])
	}
	if got["output_format"] != "mp3_high_quality" {
		t.Errorf("output_format: %#v", got["output_format"])
	}
	for _, forbidden := range []string{"aspect_ratio", "duration", "is_instrumental", "audio_format"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("music must not send %q: %#v", forbidden, got)
		}
	}
}

// WAV requests map onto the CD-quality enum rather than another bitrate knob.
func TestReplicateProviderMusicWAWFormat(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.wav")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        MusicModel,
		Prompt:       "solo piano",
		NumImages:    1,
		OutputFormat: "wav",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := (*bodies)[0]["output_format"]; got != "wav_cd_quality" {
		t.Errorf("output_format: %#v", got)
	}
	if res.Audios[0].Format != "wav" {
		t.Errorf("format: %q", res.Audios[0].Format)
	}
}

func TestReplicateProviderMusicVocalHappyPath(t *testing.T) {
	t.Run("lyrics supplied", func(t *testing.T) {
		asset := assetServer(t)
		srv, bodies := audioServer(t, asset.URL+"/song.mp3")

		p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
		res, err := p.Generate(context.Background(), &Request{
			Provider:     ProviderReplicate,
			Token:        "rtok",
			Model:        MusicVocalModel,
			Prompt:       "warm indie-folk duet",
			Lyrics:       "[Verse]\nunder the streetlight\n[Chorus]\nhold the line",
			Duration:     60,
			NumImages:    1,
			OutputFormat: "mp3",
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(res.Audios) != 1 {
			t.Fatalf("audios: %#v", res.Audios)
		}
		got := (*bodies)[0]
		if got["is_instrumental"] != false {
			t.Errorf("is_instrumental: %#v", got["is_instrumental"])
		}
		if got["lyrics"] != "[Verse]\nunder the streetlight\n[Chorus]\nhold the line" {
			t.Errorf("lyrics must pass through verbatim: %#v", got["lyrics"])
		}
		if _, ok := got["lyrics_optimizer"]; ok {
			t.Errorf("supplied lyrics must not enable the optimizer: %#v", got)
		}
		if got["audio_format"] != "mp3" || got["sample_rate"] != float64(44100) || got["bitrate"] != float64(256000) {
			t.Errorf("output knobs: %#v", got)
		}
		for _, forbidden := range []string{"duration", "music_length_ms"} {
			if _, ok := got[forbidden]; ok {
				t.Errorf("music-vocal ignores length upstream, must not send %q: %#v", forbidden, got)
			}
		}
	})
	t.Run("no lyrics lets the model write them", func(t *testing.T) {
		asset := assetServer(t)
		srv, bodies := audioServer(t, asset.URL+"/song.mp3")

		p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
		if _, err := p.Generate(context.Background(), &Request{
			Provider:     ProviderReplicate,
			Token:        "rtok",
			Model:        MusicVocalModel,
			Prompt:       "drum and bass anthem about trains",
			NumImages:    1,
			OutputFormat: "mp3",
		}); err != nil {
			t.Fatalf("Generate: %v", err)
		}
		got := (*bodies)[0]
		if got["lyrics_optimizer"] != true {
			t.Errorf("empty lyrics on a vocal track must enable the optimizer: %#v", got)
		}
		if _, ok := got["lyrics"]; ok {
			t.Errorf("no lyrics were supplied: %#v", got)
		}
	})
	t.Run("instrumental opt-in", func(t *testing.T) {
		asset := assetServer(t)
		srv, bodies := audioServer(t, asset.URL+"/inst.mp3")

		instrumental := true
		p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
		if _, err := p.Generate(context.Background(), &Request{
			Provider:     ProviderReplicate,
			Token:        "rtok",
			Model:        MusicVocalModel,
			Prompt:       "lo-fi beat",
			Instrumental: &instrumental,
			NumImages:    1,
			OutputFormat: "mp3",
		}); err != nil {
			t.Fatalf("Generate: %v", err)
		}
		got := (*bodies)[0]
		if got["is_instrumental"] != true {
			t.Errorf("is_instrumental: %#v", got["is_instrumental"])
		}
		if _, ok := got["lyrics_optimizer"]; ok {
			t.Errorf("an instrumental track needs no lyrics: %#v", got)
		}
	})
}

// Stable Audio names its length `duration`, takes a seed, and picks its own
// container — the returned one is what curds reports.
func TestReplicateProviderSFXHappyPath(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.wav")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        SFXModel,
		Prompt:       "steady heavy rain on a tin roof",
		Duration:     8,
		Seed:         7,
		NumImages:    1,
		OutputFormat: "mp3",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Audios) != 1 {
		t.Fatalf("audios: %#v", res.Audios)
	}
	if res.Audios[0].Format != "wav" {
		t.Errorf("the model's own container must be reported, got %q", res.Audios[0].Format)
	}
	got := (*bodies)[0]
	if got["duration"] != float64(8) {
		t.Errorf("duration: %#v", got["duration"])
	}
	if got["seed"] != float64(7) {
		t.Errorf("seed: %#v", got["seed"])
	}
	for _, forbidden := range []string{"music_length_ms", "force_instrumental", "output_format", "audio_format"} {
		if _, ok := got[forbidden]; ok {
			t.Errorf("sfx must not send %q: %#v", forbidden, got)
		}
	}
}

// Without -duration the SFX request still asks for the documented 10s default.
func TestReplicateProviderSFXDefaultDuration(t *testing.T) {
	asset := assetServer(t)
	srv, bodies := audioServer(t, asset.URL+"/out.mp3")

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	req := &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        SFXModel,
		Prompt:       "a single door slam",
		NumImages:    1,
		OutputFormat: "mp3",
	}
	req.applyDefaults()
	if _, err := p.Generate(context.Background(), req); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := (*bodies)[0]["duration"]; got != float64(DefaultSFXDuration) {
		t.Errorf("duration: %#v", got)
	}
}
