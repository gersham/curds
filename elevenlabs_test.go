package curds

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// elevenLabsCall is one request the ElevenLabs stub saw.
type elevenLabsCall struct {
	method string
	path   string
	query  url.Values
	key    string
	body   map[string]any
}

// elevenLabsStub stands in for api.elevenlabs.io. Audio endpoints answer with
// "audio:<output_format>" unless reject lists that format, in which case they
// answer the 403 a lower-tier plan gets. GET /v2/voices and
// /v1/shared-voices return the given voices.
type elevenLabsStub struct {
	calls   []elevenLabsCall
	reject  map[string]bool
	account []map[string]any
	library []map[string]any
}

func (s *elevenLabsStub) server(t *testing.T) *ElevenLabsProvider {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		s.calls = append(s.calls, elevenLabsCall{
			method: r.Method, path: r.URL.Path, query: r.URL.Query(),
			key: r.Header.Get("xi-api-key"), body: body,
		})
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v2/voices":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"voices": s.account, "has_more": false})
		case r.Method == http.MethodGet && r.URL.Path == "/v1/shared-voices":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"voices": s.library, "has_more": false})
		case r.Method == http.MethodPost:
			format := r.URL.Query().Get("output_format")
			if s.reject[format] {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusForbidden)
				_, _ = io.WriteString(w, `{"detail":{"type":"authorization_error","code":"subscription_required","message":"Output format '`+format+`' is only available on the Creator tier and above.","status":"output_format_not_allowed"}}`)
				return
			}
			if r.URL.Path == "/v1/music" {
				w.Header().Set("song-id", "song-123")
			}
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = io.WriteString(w, "audio:"+format)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return &ElevenLabsProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
}

func (s *elevenLabsStub) posts() []elevenLabsCall {
	var out []elevenLabsCall
	for _, c := range s.calls {
		if c.method == http.MethodPost {
			out = append(out, c)
		}
	}
	return out
}

func directTTSRequest(mut func(r *Request)) *Request {
	r := &Request{
		Provider: ProviderElevenLabs,
		Token:    "el-key",
		Model:    DefaultElevenLabsTTSModel,
		Prompt:   "[whispers] The rain had not stopped for a week.",
	}
	if mut != nil {
		mut(r)
	}
	return r
}

func TestIsElevenLabsDirectModels(t *testing.T) {
	tts := map[string]bool{
		"eleven_v4": true, "eleven_v3": true, "Eleven_Multilingual_V2 ": true, "eleven_flash_v2_5": true,
		"eleven_": false, "elevenlabs/v3": false, "eleven_v4/evil": false, "music_v2_5": false, "": false,
	}
	for in, want := range tts {
		if got := IsElevenLabsDirectTTSModel(in); got != want {
			t.Errorf("IsElevenLabsDirectTTSModel(%q) = %v want %v", in, got, want)
		}
	}
	music := map[string]bool{
		"music_v1": true, "music_v2_5": true, "Music_V2": true,
		"music_v": false, "elevenlabs/music": false, "music": false, "eleven_v4": false,
	}
	for in, want := range music {
		if got := IsElevenLabsDirectMusicModel(in); got != want {
			t.Errorf("IsElevenLabsDirectMusicModel(%q) = %v want %v", in, got, want)
		}
	}
	// Both are audio models; neither is promptless.
	for _, m := range []string{DefaultElevenLabsTTSModel, DefaultElevenLabsMusicModel} {
		if !IsAudioModel(m) || IsPromptlessModel(m) {
			t.Errorf("%s: IsAudioModel=%v IsPromptlessModel=%v", m, IsAudioModel(m), IsPromptlessModel(m))
		}
	}
	if !IsTTSModel(DefaultElevenLabsTTSModel) || IsTTSModel(DefaultElevenLabsMusicModel) {
		t.Error("eleven_v4 is TTS, music_v2_5 is not")
	}
}

func TestCheckProviderModelElevenLabs(t *testing.T) {
	for _, m := range []string{"eleven_v4", "eleven_v3", "eleven_multilingual_v2", "music_v2_5", "tts-elevenlabs", "music"} {
		if err := CheckProviderModel(ProviderElevenLabs, m); err != nil {
			t.Errorf("elevenlabs should run %s: %v", m, err)
		}
	}
	if err := CheckProviderModel(ProviderElevenLabs, "google/gemini-3.1-flash-tts"); err == nil ||
		!strings.Contains(err.Error(), "tts-elevenlabs, music") {
		t.Errorf("elevenlabs + gemini must fail with the supported list, got %v", err)
	}
	if err := CheckProviderModel(ProviderReplicate, "eleven_v4"); err == nil {
		t.Error("replicate must not accept a direct ElevenLabs model id")
	}
	if got := DefaultModel(ProviderElevenLabs); got != DefaultElevenLabsTTSModel {
		t.Errorf("elevenlabs default model: %q", got)
	}
}

func TestRequestValidateElevenLabsDirect(t *testing.T) {
	cases := []struct {
		name       string
		mut        func(r *Request)
		wantErrSub string
	}{
		{"v4 defaults", nil, ""},
		{"v4 stability and similarity", func(r *Request) {
			r.Stability = float64Ptr(0.3)
			r.SimilarityBoost = float64Ptr(0.8)
		}, ""},
		{"any voice id", func(r *Request) { r.Voice = "Ta6wSW1jA5DxUlGifuZ6" }, ""},
		{"any voice name", func(r *Request) { r.Voice = "Labhaoise" }, ""},
		{"v4 rejects style", func(r *Request) { r.Style = float64Ptr(0.5) }, "-style is not supported by eleven_v4"},
		{"v4 rejects speed", func(r *Request) { r.Speed = 1.1 }, "-speed is not supported by eleven_v4"},
		{"v4 rejects speaker boost", func(r *Request) {
			on := true
			r.SpeakerBoost = &on
		}, "-speaker-boost is not supported by eleven_v4"},
		{"v4 names the model that takes them", func(r *Request) {
			r.Style = float64Ptr(0.5)
			r.Speed = 1.1
		}, "-style and -speed are not supported by eleven_v4, which takes -stability and -similarity; add -tts-model eleven_multilingual_v2"},
		{"v3 takes stability only", func(r *Request) {
			r.Model = "eleven_v3"
			r.SimilarityBoost = float64Ptr(0.5)
		}, "-similarity is not supported by eleven_v3, which takes -stability"},
		{"multilingual v2 takes everything", func(r *Request) {
			on := false
			r.Model = "eleven_multilingual_v2"
			r.Stability = float64Ptr(0)
			r.SimilarityBoost = float64Ptr(1)
			r.Style = float64Ptr(0.4)
			r.Speed = 0.9
			r.SpeakerBoost = &on
		}, ""},
		{"unknown model passes knobs through", func(r *Request) {
			r.Model = "eleven_v5"
			r.Style = float64Ptr(0.4)
		}, ""},
		{"stability range", func(r *Request) { r.Stability = float64Ptr(1.5) }, "-stability must be 0-1"},
		{"similarity range", func(r *Request) { r.SimilarityBoost = float64Ptr(-0.1) }, "-similarity must be 0-1"},
		{"speed range", func(r *Request) {
			r.Model = "eleven_multilingual_v2"
			r.Speed = 1.5
		}, "-speed must be 0.7-1.2"},
		{"v4 text cap", func(r *Request) { r.Prompt = strings.Repeat("a", 10001) }, "eleven_v4 accepts at most 10000"},
		{"v3 text cap", func(r *Request) {
			r.Model = "eleven_v3"
			r.Prompt = strings.Repeat("a", 5001)
		}, "eleven_v3 accepts at most 5000"},
		{"rejects instructions", func(r *Request) { r.Instructions = "dry" }, "put audio tags such as [whispers] in the text"},
		{"rejects emotion", func(r *Request) { r.Emotion = "calm" }, "-emotion is only supported"},
		{"rejects a replicate provider", func(r *Request) { r.Provider = ProviderReplicate }, "does not support model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := directTTSRequest(tc.mut)
			r.applyDefaults()
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

// -similarity and -speaker-boost exist only on the direct route; Replicate's
// elevenlabs/v3 and the other TTS models reject them.
func TestDirectOnlySettingsRejectedElsewhere(t *testing.T) {
	on := true
	for _, mut := range []func(r *Request){
		func(r *Request) { r.SimilarityBoost = float64Ptr(0.5) },
		func(r *Request) { r.SpeakerBoost = &on },
	} {
		for _, model := range []string{TTSElevenLabsModel, TTSGeminiModel, TTSSpeechModel} {
			r := &Request{Provider: ProviderReplicate, Token: "t", Model: model, Prompt: "hi", NumImages: 1, OutputFormat: "mp3"}
			mut(r)
			if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "need the direct ElevenLabs route") {
				t.Errorf("%s: got %v", model, err)
			}
		}
	}
}

func TestRequestValidateElevenLabsDirectMusic(t *testing.T) {
	base := func(mut func(r *Request)) *Request {
		r := &Request{Provider: ProviderElevenLabs, Token: "k", Model: DefaultElevenLabsMusicModel, Prompt: "tense synth score"}
		if mut != nil {
			mut(r)
		}
		r.applyDefaults()
		return r
	}
	cases := []struct {
		name       string
		mut        func(r *Request)
		wantErrSub string
	}{
		{"defaults", nil, ""},
		{"3s floor", func(r *Request) { r.Duration = 3 }, ""},
		{"600s ceiling", func(r *Request) { r.Duration = 600 }, ""},
		{"below the floor", func(r *Request) { r.Duration = 2 }, "3-600 seconds"},
		{"above the ceiling", func(r *Request) { r.Duration = 601 }, "3-600 seconds"},
		{"no lyrics", func(r *Request) { r.Lyrics = "la la" }, "-lyrics is only supported"},
		{"no seed", func(r *Request) { r.Seed = 7 }, "no seed input"},
		{"wav allowed", func(r *Request) { r.OutputFormat = "wav" }, ""},
		{"needs the elevenlabs provider", func(r *Request) { r.Provider = ProviderReplicate }, "does not support model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := base(tc.mut).Validate()
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
	r := base(nil)
	if r.Instrumental == nil || !*r.Instrumental {
		t.Errorf("direct music defaults to instrumental like -model music, got %v", r.Instrumental)
	}
	if r.OutputFormat != "mp3" || r.AspectRatio != "" {
		t.Errorf("audio defaults: format=%q ratio=%q", r.OutputFormat, r.AspectRatio)
	}
}

// The request the direct route sends: the voice id in the path, the key in
// xi-api-key, the model_id and text in the body, only the knobs that were set
// in voice_settings, and the best mp3 the format ladder offers.
func TestElevenLabsSpeechRequestShape(t *testing.T) {
	stub := &elevenLabsStub{}
	c := &Client{ElevenLabs: stub.server(t)}
	res, err := c.Generate(context.Background(), directTTSRequest(func(r *Request) {
		r.Voice = "Ta6wSW1jA5DxUlGifuZ6"
		r.Stability = float64Ptr(0)
		r.SimilarityBoost = float64Ptr(0.8)
	}))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	posts := stub.posts()
	if len(posts) != 1 || len(stub.calls) != 1 {
		t.Fatalf("an id needs no voice lookup and one POST, got %d calls", len(stub.calls))
	}
	call := posts[0]
	if call.path != "/v1/text-to-speech/Ta6wSW1jA5DxUlGifuZ6" {
		t.Errorf("path: %s", call.path)
	}
	if call.key != "el-key" {
		t.Errorf("xi-api-key: %q", call.key)
	}
	if got := call.query.Get("output_format"); got != "mp3_44100_192" {
		t.Errorf("output_format: %q", got)
	}
	if call.body["model_id"] != "eleven_v4" || call.body["text"] != "[whispers] The rain had not stopped for a week." {
		t.Errorf("body: %#v", call.body)
	}
	settings, _ := call.body["voice_settings"].(map[string]any)
	if settings["stability"] != float64(0) || settings["similarity_boost"] != 0.8 {
		t.Errorf("voice_settings (0 stability must survive): %#v", settings)
	}
	for _, k := range []string{"style", "speed", "use_speaker_boost"} {
		if _, ok := settings[k]; ok {
			t.Errorf("unset %s must be omitted: %#v", k, settings)
		}
	}
	if len(res.Audios) != 1 || string(res.Audios[0].Bytes) != "audio:mp3_44100_192" || res.Audios[0].Format != "mp3" {
		t.Errorf("result: %#v", res.Audios)
	}
}

func TestElevenLabsSpeechDefaultsAndAllKnobs(t *testing.T) {
	stub := &elevenLabsStub{}
	c := &Client{ElevenLabs: stub.server(t)}
	if _, err := c.Generate(context.Background(), directTTSRequest(nil)); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	call := stub.posts()[0]
	if call.path != "/v1/text-to-speech/"+DefaultElevenLabsVoiceID {
		t.Errorf("default voice path: %s", call.path)
	}
	if _, ok := call.body["voice_settings"]; ok {
		t.Errorf("no knobs set means no voice_settings, got %#v", call.body)
	}

	stub2 := &elevenLabsStub{}
	c2 := &Client{ElevenLabs: stub2.server(t)}
	off := false
	if _, err := c2.Generate(context.Background(), directTTSRequest(func(r *Request) {
		r.Model = "eleven_multilingual_v2"
		r.Stability = float64Ptr(0.5)
		r.SimilarityBoost = float64Ptr(0.75)
		r.Style = float64Ptr(0.2)
		r.Speed = 1.1
		r.SpeakerBoost = &off
	})); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	settings, _ := stub2.posts()[0].body["voice_settings"].(map[string]any)
	want := map[string]any{"stability": 0.5, "similarity_boost": 0.75, "style": 0.2, "speed": 1.1, "use_speaker_boost": false}
	for k, v := range want {
		if settings[k] != v {
			t.Errorf("voice_settings[%s] = %#v want %#v (all: %#v)", k, settings[k], v, settings)
		}
	}
}

// A plan below Creator answers 403 output_format_not_allowed for 192 kbps;
// curds steps down to 128 kbps instead of failing.
func TestElevenLabsFormatFallback(t *testing.T) {
	stub := &elevenLabsStub{reject: map[string]bool{"mp3_44100_192": true}}
	c := &Client{ElevenLabs: stub.server(t)}
	res, err := c.Generate(context.Background(), directTTSRequest(nil))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	posts := stub.posts()
	if len(posts) != 2 || posts[0].query.Get("output_format") != "mp3_44100_192" || posts[1].query.Get("output_format") != "mp3_44100_128" {
		t.Fatalf("format ladder: %#v", posts)
	}
	if string(res.Audios[0].Bytes) != "audio:mp3_44100_128" || res.Audios[0].Format != "mp3" {
		t.Errorf("result: %#v", res.Audios[0])
	}

	// wav: 44.1 kHz needs Pro, so a Starter plan gets 24 kHz wav.
	stubWav := &elevenLabsStub{reject: map[string]bool{"wav_44100": true}}
	cw := &Client{ElevenLabs: stubWav.server(t)}
	res, err = cw.Generate(context.Background(), directTTSRequest(func(r *Request) { r.OutputFormat = "wav" }))
	if err != nil {
		t.Fatalf("Generate wav: %v", err)
	}
	if res.Audios[0].Format != "wav" || string(res.Audios[0].Bytes) != "audio:wav_24000" {
		t.Errorf("wav fallback: %#v", res.Audios[0])
	}

	// Every rung refused: the last error surfaces with the API's message.
	stubAll := &elevenLabsStub{reject: map[string]bool{"mp3_44100_192": true, "mp3_44100_128": true}}
	ca := &Client{ElevenLabs: stubAll.server(t)}
	_, err = ca.Generate(context.Background(), directTTSRequest(nil))
	if err == nil || !strings.Contains(err.Error(), "elevenlabs API 403: Output format 'mp3_44100_128'") {
		t.Errorf("exhausted ladder: %v", err)
	}
}

// Errors that are not about the output format are returned at once, with the
// API's message, and never retried.
func TestElevenLabsAPIErrorNoRetry(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"detail":{"type":"authentication_error","code":"invalid_api_key","message":"Invalid API key"}}`)
	}))
	defer srv.Close()
	c := &Client{ElevenLabs: &ElevenLabsProvider{HTTPClient: srv.Client(), APIBase: srv.URL}}
	_, err := c.Generate(context.Background(), directTTSRequest(func(r *Request) { r.Voice = DefaultElevenLabsVoiceID }))
	if err == nil || err.Error() != "elevenlabs API 401: Invalid API key" {
		t.Errorf("error: %v", err)
	}
	if calls != 1 {
		t.Errorf("auth errors must not walk the format ladder, got %d calls", calls)
	}
}

func TestParseElevenLabsErrorShapes(t *testing.T) {
	cases := map[string]string{
		`{"detail":{"message":"bad voice","status":"voice_not_found"}}`: "bad voice",
		`{"detail":[{"loc":["body","text"],"msg":"field required"}]}`:   "field required",
		`{"detail":"plain detail"}`:                                     "plain detail",
	}
	for body, want := range cases {
		if got := parseElevenLabsError([]byte(body)).text([]byte(body)); got != want {
			t.Errorf("%s: got %q want %q", body, got, want)
		}
	}
	if got := parseElevenLabsError([]byte("gateway down")).text([]byte("gateway down")); got != "gateway down" {
		t.Errorf("non-JSON body: %q", got)
	}
}

var accountVoicesFixture = []map[string]any{
	{"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George - Warm, Captivating Storyteller", "category": "premade",
		"description": "Warm resonance that instantly captivates listeners.",
		"labels":      map[string]any{"accent": "british", "age": "middle_aged", "gender": "male"}},
	{"voice_id": "e1FTlhfUIxc3b0ztuqSK", "name": "Brian", "category": "generated",
		"labels": map[string]any{"accent": "british", "age": "old", "gender": "male", "descriptive": "mature"}},
	{"voice_id": "nPczCjzI2devNBz1zQrb", "name": "Brian - Deep, Resonant and Comforting", "category": "premade",
		"labels": map[string]any{"accent": "american", "gender": "male"}},
	{"voice_id": "aaaaaaaaaaaaaaaaaaa1", "name": "Sam - One", "labels": map[string]any{}},
	{"voice_id": "aaaaaaaaaaaaaaaaaaa2", "name": "Sam - Two", "labels": map[string]any{}},
}

// A name resolves against the account's voices (GET /v2/voices?search=…): the
// part before " - " matches, an exact full name beats that, several equal
// matches are ambiguous, and a miss names curds voices.
func TestElevenLabsVoiceNameResolution(t *testing.T) {
	cases := []struct {
		voice      string
		wantID     string
		wantErrSub string
	}{
		{"george", "JBFqnCBsd6RMkjVDRZzb", ""},
		{"Brian", "e1FTlhfUIxc3b0ztuqSK", ""},
		{"Brian - Deep, Resonant and Comforting", "nPczCjzI2devNBz1zQrb", ""},
		{"Sam", "", `voice "Sam" matches 2 voices`},
		{"Rachel", "", `no voice named "Rachel" in your ElevenLabs account; find one with curds voices -search "Rachel"`},
	}
	for _, tc := range cases {
		t.Run(tc.voice, func(t *testing.T) {
			stub := &elevenLabsStub{account: accountVoicesFixture}
			c := &Client{ElevenLabs: stub.server(t)}
			_, err := c.Generate(context.Background(), directTTSRequest(func(r *Request) { r.Voice = tc.voice }))
			lookup := stub.calls[0]
			if lookup.method != http.MethodGet || lookup.path != "/v2/voices" || lookup.query.Get("search") != tc.voice || lookup.key != "el-key" {
				t.Errorf("lookup: %#v", lookup)
			}
			if tc.wantErrSub != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
					t.Fatalf("got %v, want substring %q", err, tc.wantErrSub)
				}
				if len(stub.posts()) != 0 {
					t.Error("an unresolved voice must not reach the TTS endpoint")
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if got := stub.posts()[0].path; got != "/v1/text-to-speech/"+tc.wantID {
				t.Errorf("resolved path: %s", got)
			}
		})
	}
}

func TestElevenLabsVoiceListings(t *testing.T) {
	stub := &elevenLabsStub{
		account: accountVoicesFixture[:2],
		library: []map[string]any{{
			"voice_id": "3XD5qTvPWht0mpTnYXY0", "name": "Lily - Warm British Audiobook Narrator", "accent": "british",
			"age": "middle_aged", "gender": "female", "descriptive": "classy", "category": "high_quality",
			"description": "Warm and classy British\naudiobook narrator.",
		}},
	}
	p := stub.server(t)
	q := VoiceQuery{Search: "narrator", Accent: "british", Gender: "female", Age: "middle_aged", Limit: 5}

	account, err := p.AccountVoices(context.Background(), "el-key", q)
	if err != nil {
		t.Fatalf("AccountVoices: %v", err)
	}
	library, err := p.LibraryVoices(context.Background(), "el-key", q)
	if err != nil {
		t.Fatalf("LibraryVoices: %v", err)
	}
	for i, path := range []string{"/v2/voices", "/v1/shared-voices"} {
		call := stub.calls[i]
		if call.path != path || call.key != "el-key" {
			t.Errorf("call %d: %s key=%q", i, call.path, call.key)
		}
		for k, v := range map[string]string{"search": "narrator", "accent": "british", "gender": "female", "age": "middle_aged", "page_size": "5"} {
			if got := call.query.Get(k); got != v {
				t.Errorf("%s %s=%q want %q", path, k, got, v)
			}
		}
	}
	if len(account) != 2 || account[0].ID != "JBFqnCBsd6RMkjVDRZzb" || account[0].Accent != "british" ||
		account[0].Gender != "male" || account[0].Age != "middle_aged" || account[0].Source != "account" {
		t.Errorf("account voices: %#v", account)
	}
	if account[1].Description != "mature" {
		t.Errorf("an account voice without a description falls back to its descriptive label: %q", account[1].Description)
	}
	if len(library) != 1 || library[0].Source != "library" || library[0].Gender != "female" {
		t.Errorf("library voices: %#v", library)
	}
	line := library[0].String()
	for _, want := range []string{
		"source=library", "id=3XD5qTvPWht0mpTnYXY0", `name="Lily - Warm British Audiobook Narrator"`,
		"accent=british", "age=middle_aged", "gender=female",
		`description="Warm and classy British audiobook narrator."`,
	} {
		if !strings.Contains(line, want) {
			t.Errorf("voice line %q missing %q", line, want)
		}
	}
	if (VoiceQuery{}).values().Get("page_size") != "30" || (VoiceQuery{Limit: 500}).values().Get("page_size") != "100" {
		t.Error("page_size defaults to 30 and caps at 100")
	}
}

// Direct music goes to POST /v1/music with the newest music model, an exact
// music_length_ms, force_instrumental, and the best mp3 rung.
func TestElevenLabsMusicCompose(t *testing.T) {
	stub := &elevenLabsStub{reject: map[string]bool{"mp3_48000_320": true}}
	c := &Client{ElevenLabs: stub.server(t)}
	res, err := c.Generate(context.Background(), &Request{
		Provider: ProviderElevenLabs, Token: "el-key", Model: DefaultElevenLabsMusicModel,
		Prompt: "tense minimal synth score", Duration: 45,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	posts := stub.posts()
	if len(posts) != 2 || posts[0].path != "/v1/music" || posts[1].query.Get("output_format") != "mp3_48000_192" {
		t.Fatalf("music calls: %#v", posts)
	}
	body := posts[1].body
	if body["prompt"] != "tense minimal synth score" || body["model_id"] != "music_v2_5" {
		t.Errorf("body: %#v", body)
	}
	if body["music_length_ms"] != float64(45000) || body["force_instrumental"] != true {
		t.Errorf("length/instrumental: %#v", body)
	}
	if posts[1].key != "el-key" {
		t.Errorf("xi-api-key: %q", posts[1].key)
	}
	if res.Audios[0].Format != "mp3" || string(res.Audios[0].Bytes) != "audio:mp3_48000_192" {
		t.Errorf("result: %#v", res.Audios[0])
	}

	// Vocals, no length, wav requested: the music endpoint has no wav, so the
	// mp3 comes back labelled mp3 for the CLI to transcode.
	stub2 := &elevenLabsStub{}
	c2 := &Client{ElevenLabs: stub2.server(t)}
	vocals := false
	res, err = c2.Generate(context.Background(), &Request{
		Provider: ProviderElevenLabs, Token: "el-key", Model: DefaultElevenLabsMusicModel,
		Prompt: "indie folk duet", Instrumental: &vocals, OutputFormat: "wav",
	})
	if err != nil {
		t.Fatalf("Generate vocals: %v", err)
	}
	body = stub2.posts()[0].body
	if body["force_instrumental"] != false {
		t.Errorf("-instrumental=false: %#v", body)
	}
	if _, ok := body["music_length_ms"]; ok {
		t.Errorf("no -duration means the model picks the length: %#v", body)
	}
	if got := stub2.posts()[0].query.Get("output_format"); got != "mp3_48000_320" {
		t.Errorf("wav still asks for the best mp3: %q", got)
	}
	if res.Audios[0].Format != "mp3" {
		t.Errorf("music container: %q", res.Audios[0].Format)
	}
}
