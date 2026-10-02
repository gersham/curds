package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gersham/curds"
)

// elevenLabsCLICall is one request the CLI's ElevenLabs stub saw.
type elevenLabsCLICall struct {
	method string
	path   string
	query  url.Values
	key    string
	body   map[string]any
}

// elevenLabsCLIStub stands in for api.elevenlabs.io: audio POSTs answer
// "el-audio" and GET /v2/voices / /v1/shared-voices answer one voice each.
func elevenLabsCLIStub(t *testing.T) (*curds.ElevenLabsProvider, *[]elevenLabsCLICall) {
	t.Helper()
	calls := &[]elevenLabsCLICall{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		*calls = append(*calls, elevenLabsCLICall{r.Method, r.URL.Path, r.URL.Query(), r.Header.Get("xi-api-key"), body})
		switch r.URL.Path {
		case "/v2/voices":
			_ = json.NewEncoder(w).Encode(map[string]any{"voices": []map[string]any{{
				"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George - Warm, Captivating Storyteller",
				"description": "Warm resonance.",
				"labels":      map[string]any{"accent": "british", "age": "middle_aged", "gender": "male"},
			}}})
		case "/v1/shared-voices":
			_ = json.NewEncoder(w).Encode(map[string]any{"voices": []map[string]any{{
				"voice_id": "3XD5qTvPWht0mpTnYXY0", "name": "Lily - Warm British Audiobook Narrator",
				"accent": "british", "age": "middle_aged", "gender": "female", "description": "Warm and classy.",
			}}})
		default:
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = io.WriteString(w, "el-audio")
		}
	}))
	t.Cleanup(srv.Close)
	return &curds.ElevenLabsProvider{HTTPClient: srv.Client(), APIBase: srv.URL}, calls
}

func elevenLabsPosts(calls []elevenLabsCLICall) []elevenLabsCLICall {
	var out []elevenLabsCLICall
	for _, c := range calls {
		if c.method == http.MethodPost {
			out = append(out, c)
		}
	}
	return out
}

// runRealMainLogged is runRealMain that also returns the logfmt events, so a
// test can read event=elevenlabs.route.
func runRealMainLogged(t *testing.T, args []string) (string, string, error) {
	t.Helper()
	oldArgs, oldCmd := os.Args, flag.CommandLine
	flag.CommandLine = flag.NewFlagSet("curds", flag.ContinueOnError)
	os.Args = append([]string{"curds"}, args...)
	t.Cleanup(func() { os.Args, flag.CommandLine = oldArgs, oldCmd })
	var logs bytes.Buffer
	var err error
	printed := captureStdout(t, func() { err = realMain(newLogger(&logs), time.Now()) })
	return printed, logs.String(), err
}

// writeConfigTokens writes a config file at $CURDS_CONFIG with the given
// [tokens] lines (ttsTestEnv already pointed CURDS_CONFIG at a temp path).
func writeConfigTokens(t *testing.T, lines string) {
	t.Helper()
	if err := os.WriteFile(os.Getenv("CURDS_CONFIG"), []byte("[tokens]\n"+lines+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

// With an elevenlabs token, -model tts-elevenlabs and -model music go to the
// ElevenLabs API; without one they stay on Replicate. Both stubs are live in
// every case, so a mis-route is caught.
func TestElevenLabsRouteSelection(t *testing.T) {
	type stubs struct {
		repInputs *[]map[string]any
		el        *[]elevenLabsCLICall
	}
	setup := func(t *testing.T) stubs {
		t.Helper()
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		rep, repInputs := replicateAudioStub(t, "mp3")
		el, calls := elevenLabsCLIStub(t)
		withClient(t, &curds.Client{Replicate: rep, ElevenLabs: el})
		return stubs{repInputs, calls}
	}
	out := func(t *testing.T) string { return filepath.Join(t.TempDir(), "line.mp3") }

	t.Run("ELEVENLABS_API_KEY sends tts-elevenlabs direct with eleven_v4", func(t *testing.T) {
		s := setup(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		path := out(t)
		_, logs, err := runRealMainLogged(t, []string{
			"-no-tui", "-model", "tts-elevenlabs", "-voice", "Ta6wSW1jA5DxUlGifuZ6",
			"-stability", "0.4", "-similarity", "0.9",
			"-prompt", "[whispers] It was a very long day.", "-output", path,
		})
		if err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*s.repInputs) != 0 {
			t.Errorf("the direct route must not touch Replicate: %#v", *s.repInputs)
		}
		posts := elevenLabsPosts(*s.el)
		if len(posts) != 1 {
			t.Fatalf("elevenlabs posts: %#v", *s.el)
		}
		call := posts[0]
		if call.path != "/v1/text-to-speech/Ta6wSW1jA5DxUlGifuZ6" || call.key != "el-env" {
			t.Errorf("request: %s key=%q", call.path, call.key)
		}
		if call.body["model_id"] != curds.DefaultElevenLabsTTSModel || call.body["text"] != "[whispers] It was a very long day." {
			t.Errorf("body: %#v", call.body)
		}
		settings, _ := call.body["voice_settings"].(map[string]any)
		if settings["stability"] != 0.4 || settings["similarity_boost"] != 0.9 {
			t.Errorf("voice_settings: %#v", settings)
		}
		if !strings.Contains(logs, "event=elevenlabs.route route=direct model=eleven_v4") || !strings.Contains(logs, `reason="elevenlabs token"`) {
			t.Errorf("route log missing:\n%s", logs)
		}
		if data, err := os.ReadFile(path); err != nil || string(data) != "el-audio" {
			t.Errorf("saved file: %q err=%v", data, err)
		}
	})

	t.Run("a [tokens] elevenlabs entry in config also routes direct", func(t *testing.T) {
		s := setup(t)
		writeConfigTokens(t, `elevenlabs = "el-config"`)
		if _, err := runRealMain(t, []string{"-no-tui", "-model", "tts-elevenlabs", "-prompt", "Hello.", "-output", out(t)}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		posts := elevenLabsPosts(*s.el)
		if len(posts) != 1 || posts[0].key != "el-config" || len(*s.repInputs) != 0 {
			t.Fatalf("config key: posts=%#v replicate=%d", posts, len(*s.repInputs))
		}
		if posts[0].path != "/v1/text-to-speech/"+curds.DefaultElevenLabsVoiceID {
			t.Errorf("no -voice means the default voice id: %s", posts[0].path)
		}
	})

	t.Run("an ELEVENLABS_API_KEY line in .env also routes direct", func(t *testing.T) {
		s := setup(t)
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("ELEVENLABS_API_KEY=el-dotenv\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		t.Chdir(dir)
		if _, err := runRealMain(t, []string{"-no-tui", "-model", "tts-elevenlabs", "-prompt", "Hello.", "-output", out(t)}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		posts := elevenLabsPosts(*s.el)
		if len(posts) != 1 || posts[0].key != "el-dotenv" {
			t.Fatalf(".env key: %#v", posts)
		}
	})

	t.Run("no elevenlabs token keeps tts-elevenlabs on Replicate", func(t *testing.T) {
		s := setup(t)
		_, logs, err := runRealMainLogged(t, []string{"-no-tui", "-model", "tts-elevenlabs", "-voice", "Rachel", "-prompt", "Fallback.", "-output", out(t)})
		if err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*s.el) != 0 {
			t.Errorf("no token must not reach ElevenLabs: %#v", *s.el)
		}
		if len(*s.repInputs) != 1 || (*s.repInputs)[0]["voice"] != "Rachel" || (*s.repInputs)[0]["prompt"] != "Fallback." {
			t.Fatalf("replicate input: %#v", *s.repInputs)
		}
		if !strings.Contains(logs, "event=elevenlabs.route route=replicate model=elevenlabs/v3") || !strings.Contains(logs, `reason="no elevenlabs token"`) {
			t.Errorf("route log missing:\n%s", logs)
		}
	})

	t.Run("-provider replicate keeps Replicate even with a token", func(t *testing.T) {
		s := setup(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		_, logs, err := runRealMainLogged(t, []string{"-no-tui", "-provider", "replicate", "-model", "tts-elevenlabs", "-prompt", "Hi.", "-output", out(t)})
		if err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*s.el) != 0 || len(*s.repInputs) != 1 {
			t.Fatalf("forced replicate: el=%d replicate=%d", len(*s.el), len(*s.repInputs))
		}
		if !strings.Contains(logs, `reason="-provider replicate"`) {
			t.Errorf("route log missing:\n%s", logs)
		}
	})

	t.Run("-tts-model picks the ElevenLabs model id", func(t *testing.T) {
		s := setup(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "tts-elevenlabs", "-tts-model", "eleven_multilingual_v2",
			"-style", "0.3", "-speed", "1.1", "-speaker-boost", "-prompt", "Hi.", "-output", out(t),
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		call := elevenLabsPosts(*s.el)[0]
		if call.body["model_id"] != "eleven_multilingual_v2" {
			t.Errorf("model_id: %#v", call.body["model_id"])
		}
		settings, _ := call.body["voice_settings"].(map[string]any)
		if settings["style"] != 0.3 || settings["speed"] != 1.1 || settings["use_speaker_boost"] != true {
			t.Errorf("voice_settings: %#v", settings)
		}
	})

	t.Run("a voice name on the direct route resolves through GET /v2/voices", func(t *testing.T) {
		s := setup(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		if _, err := runRealMain(t, []string{"-no-tui", "-model", "tts-elevenlabs", "-voice", "George", "-prompt", "Hi.", "-output", out(t)}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if (*s.el)[0].path != "/v2/voices" || (*s.el)[0].query.Get("search") != "George" {
			t.Errorf("lookup: %#v", (*s.el)[0])
		}
		if got := elevenLabsPosts(*s.el)[0].path; got != "/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb" {
			t.Errorf("resolved path: %s", got)
		}
	})

	t.Run("-model music goes to POST /v1/music with a token", func(t *testing.T) {
		s := setup(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		_, logs, err := runRealMainLogged(t, []string{"-no-tui", "-model", "music", "-duration", "400", "-prompt", "slow synth score", "-output", out(t)})
		if err != nil {
			t.Fatalf("realMain: %v", err)
		}
		posts := elevenLabsPosts(*s.el)
		if len(posts) != 1 || posts[0].path != "/v1/music" || len(*s.repInputs) != 0 {
			t.Fatalf("music route: %#v replicate=%d", posts, len(*s.repInputs))
		}
		body := posts[0].body
		if body["model_id"] != curds.DefaultElevenLabsMusicModel || body["music_length_ms"] != float64(400000) || body["force_instrumental"] != true {
			t.Errorf("music body: %#v", body)
		}
		if !strings.Contains(logs, "event=elevenlabs.route route=direct model=music_v2_5") {
			t.Errorf("route log missing:\n%s", logs)
		}
	})

	t.Run("-model music without a token stays on Replicate elevenlabs/music", func(t *testing.T) {
		s := setup(t)
		if _, err := runRealMain(t, []string{"-no-tui", "-model", "music", "-duration", "45", "-instrumental=false", "-prompt", "folk duet", "-output", out(t)}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*s.el) != 0 || len(*s.repInputs) != 1 {
			t.Fatalf("music fallback: el=%d replicate=%d", len(*s.el), len(*s.repInputs))
		}
		in := (*s.repInputs)[0]
		if in["music_length_ms"] != float64(45000) || in["force_instrumental"] != false || in["output_format"] != "mp3_high_quality" {
			t.Errorf("replicate music input: %#v", in)
		}
	})
}

// Route-dependent flag problems are usage errors (exit 2), decided before any
// network call.
func TestElevenLabsFlagsExitTwo(t *testing.T) {
	cases := []struct {
		name string
		key  bool
		args []string
		want string
	}{
		{"-tts-model without a token", false, []string{"-model", "tts-elevenlabs", "-tts-model", "eleven_v3"}, "-tts-model needs the direct ElevenLabs route"},
		{"-tts-model with -provider replicate", true, []string{"-provider", "replicate", "-model", "tts-elevenlabs", "-tts-model", "eleven_v3"}, "drop -provider replicate"},
		{"-tts-model on another model", true, []string{"-model", "tts", "-tts-model", "eleven_v3"}, "-tts-model is only supported by -model tts-elevenlabs"},
		{"-tts-model on music", true, []string{"-model", "music", "-tts-model", "eleven_v3"}, "not -model music"},
		{"-tts-model that is not an ElevenLabs id", true, []string{"-model", "tts-elevenlabs", "-tts-model", "gpt-4o-mini-tts"}, "is not an ElevenLabs text-to-speech model id"},
		{"-style on eleven_v4", true, []string{"-model", "tts-elevenlabs", "-style", "0.5"}, "-style is not supported by eleven_v4"},
		{"-speed on eleven_v3", true, []string{"-model", "tts-elevenlabs", "-tts-model", "eleven_v3", "-speed", "1.1"}, "-speed is not supported by eleven_v3"},
		{"-similarity on Replicate", false, []string{"-model", "tts-elevenlabs", "-similarity", "0.5"}, "need the direct ElevenLabs route"},
		{"-speaker-boost on Gemini", true, []string{"-model", "tts", "-speaker-boost"}, "need the direct ElevenLabs route"},
		{"-instructions on the direct route", true, []string{"-model", "tts-elevenlabs", "-instructions", "dry"}, "put audio tags such as [whispers] in the text"},
		{"-similarity out of range", true, []string{"-model", "tts-elevenlabs", "-similarity", "1.5"}, "-similarity must be 0-1"},
		{"direct music above 600s", true, []string{"-model", "music", "-duration", "700"}, "3-600 seconds"},
		{"Replicate music above 300s", false, []string{"-model", "music", "-duration", "400"}, "5-300 seconds"},
		{"Replicate voice enum still applies", false, []string{"-model", "tts-elevenlabs", "-voice", "Bob"}, "any voice id works on the direct ElevenLabs route"},
		{"direct route without a token", false, []string{"-provider", "elevenlabs", "-model", "tts-elevenlabs"}, "no elevenlabs token available"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ttsTestEnv(t)
			t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
			if tc.key {
				t.Setenv("ELEVENLABS_API_KEY", "el-env")
			}
			el, calls := elevenLabsCLIStub(t)
			withClient(t, &curds.Client{ElevenLabs: el})
			args := append([]string{"-no-tui", "-prompt", "hi", "-output", filepath.Join(t.TempDir(), "x.mp3")}, tc.args...)
			_, err := runRealMain(t, args)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want substring %q", err, tc.want)
			}
			var ue *usageError
			if !errors.As(err, &ue) && !strings.Contains(tc.want, "no elevenlabs token") {
				t.Errorf("must exit 2, got %T", err)
			}
			if len(*calls) != 0 {
				t.Errorf("no request may be sent: %#v", *calls)
			}
		})
	}
}

// `curds voices` lists account voices and searches the library with the same
// filters, one logfmt line per voice on stdout.
func TestVoicesSubcommand(t *testing.T) {
	withVoices := func(t *testing.T) *[]elevenLabsCLICall {
		t.Helper()
		el, calls := elevenLabsCLIStub(t)
		old := newVoicesProvider
		newVoicesProvider = func() *curds.ElevenLabsProvider { return el }
		t.Cleanup(func() { newVoicesProvider = old })
		return calls
	}
	run := func(t *testing.T, args ...string) (string, error) {
		t.Helper()
		var err error
		out := captureStdout(t, func() { err = realMainVoices(newLogger(io.Discard), time.Now(), args) })
		return out, err
	}

	t.Run("both sources with filters", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		calls := withVoices(t)
		out, err := run(t, "-search", "narrator", "-accent", "british", "-gender", "female", "-limit", "10")
		if err != nil {
			t.Fatalf("voices: %v", err)
		}
		if len(*calls) != 2 || (*calls)[0].path != "/v2/voices" || (*calls)[1].path != "/v1/shared-voices" {
			t.Fatalf("calls: %#v", *calls)
		}
		for _, c := range *calls {
			if c.key != "el-env" || c.query.Get("search") != "narrator" || c.query.Get("accent") != "british" ||
				c.query.Get("gender") != "female" || c.query.Get("page_size") != "10" {
				t.Errorf("%s query=%v key=%q", c.path, c.query, c.key)
			}
		}
		lines := strings.Split(strings.TrimSpace(out), "\n")
		if len(lines) != 2 {
			t.Fatalf("stdout: %q", out)
		}
		for _, want := range []string{"source=account id=JBFqnCBsd6RMkjVDRZzb", "accent=british age=middle_aged gender=male", `description="Warm resonance."`} {
			if !strings.Contains(lines[0], want) {
				t.Errorf("account line %q missing %q", lines[0], want)
			}
		}
		if !strings.HasPrefix(lines[1], "source=library id=3XD5qTvPWht0mpTnYXY0") || !strings.Contains(lines[1], "gender=female") {
			t.Errorf("library line: %q", lines[1])
		}
	})

	t.Run("-source account skips the library", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("ELEVENLABS_API_KEY", "el-env")
		calls := withVoices(t)
		if _, err := run(t, "-source", "account"); err != nil {
			t.Fatalf("voices: %v", err)
		}
		if len(*calls) != 1 || (*calls)[0].path != "/v2/voices" {
			t.Errorf("calls: %#v", *calls)
		}
	})

	t.Run("no token", func(t *testing.T) {
		ttsTestEnv(t)
		calls := withVoices(t)
		_, err := run(t)
		if err == nil || !strings.Contains(err.Error(), "no elevenlabs token available") || len(*calls) != 0 {
			t.Errorf("got %v calls=%d", err, len(*calls))
		}
	})

	for _, args := range [][]string{{"-source", "web"}, {"-limit", "0"}, {"narrator"}} {
		t.Run("usage "+strings.Join(args, " "), func(t *testing.T) {
			ttsTestEnv(t)
			t.Setenv("ELEVENLABS_API_KEY", "el-env")
			withVoices(t)
			_, err := run(t, args...)
			var ue *usageError
			if !errors.As(err, &ue) {
				t.Errorf("want a usage error, got %v", err)
			}
		})
	}
}

// The help is the contract users and agents read.
func TestHelpTextMentionsElevenLabsDirect(t *testing.T) {
	help := helpText()
	for _, want := range []string{
		"curds voices",
		"ELEVENLABS_API_KEY",
		"-tts-model ID",
		"-similarity N",
		"-speaker-boost",
		"eleven_v4",
		"music_v2_5",
		"POST /v1/text-to-speech/{voice_id}",
		"POST /v1/music",
		"-provider replicate",
		"audio tags",
	} {
		if !strings.Contains(help, want) {
			t.Errorf("help is missing %q", want)
		}
	}
	vhelp := voicesHelpText()
	for _, want := range []string{"-search TEXT", "-accent", "-gender", "GET /v2/voices", "GET /v1/shared-voices"} {
		if !strings.Contains(vhelp, want) {
			t.Errorf("voices help is missing %q", want)
		}
	}
}
