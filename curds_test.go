package curds

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAutoDetectProvider(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
		want string
	}{
		{"openai preferred over replicate", map[string]string{"OPENAI_API_KEY": "x", "REPLICATE_API_TOKEN": "y"}, ProviderOpenAI},
		{"only replicate", map[string]string{"REPLICATE_API_TOKEN": "y"}, ProviderReplicate},
		{"only openai", map[string]string{"OPENAI_API_KEY": "x"}, ProviderOpenAI},
		{"none", map[string]string{}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AutoDetectProvider(func(k string) string { return tc.env[k] })
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestResolveSize(t *testing.T) {
	cases := []struct {
		name  string
		req   Request
		want  string
	}{
		{"explicit size wins", Request{Size: "1920x1088", AspectRatio: "1:1"}, "1920x1088"},
		{"16:9 maps to ~1080p+", Request{AspectRatio: "16:9"}, "2048x1152"},
		{"9:16 portrait", Request{AspectRatio: "9:16"}, "1152x2048"},
		{"unknown ratio falls back to auto", Request{AspectRatio: "weird"}, "auto"},
		{"4k landscape", Request{AspectRatio: "16:9-4k"}, "3840x2160"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveSize(&tc.req); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRequestValidate(t *testing.T) {
	base := Request{
		Provider:     ProviderOpenAI,
		Token:        "tok",
		Prompt:       "hi",
		NumImages:    1,
		OutputFormat: "webp",
	}

	cases := []struct {
		name  string
		mut   func(r *Request)
		wantErrSub string
	}{
		{"valid", func(r *Request) {}, ""},
		{"missing prompt", func(r *Request) { r.Prompt = "" }, "prompt"},
		{"missing token", func(r *Request) { r.Token = "" }, "token"},
		{"bad provider", func(r *Request) { r.Provider = "claude" }, "unsupported provider"},
		{"too many images", func(r *Request) { r.NumImages = 20 }, "num_images"},
		{"too few images", func(r *Request) { r.NumImages = 0 }, "num_images"},
		{"bad output format", func(r *Request) { r.OutputFormat = "tiff" }, "output_format"},
		{"bad compression", func(r *Request) { r.OutputCompression = 200 }, "output_compression"},
		{"too many input images", func(r *Request) {
			r.InputImages = make([]string, MaxInputImages+1)
		}, "input images"},
		{"replicate rejects 16:9", func(r *Request) {
			r.Provider = ProviderReplicate
			r.AspectRatio = "16:9"
		}, "replicate only supports"},
		{"replicate accepts 3:2", func(r *Request) {
			r.Provider = ProviderReplicate
			r.AspectRatio = "3:2"
		}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			tc.mut(&r)
			err := r.Validate()
			if tc.wantErrSub == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErrSub)
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("expected error containing %q, got %q", tc.wantErrSub, err.Error())
			}
		})
	}
}

func TestRequestApplyDefaults(t *testing.T) {
	r := Request{Provider: ProviderOpenAI}
	r.applyDefaults()
	if r.Model != DefaultOpenAIModel {
		t.Errorf("Model default: got %q", r.Model)
	}
	if r.NumImages != 1 {
		t.Errorf("NumImages default: got %d", r.NumImages)
	}
	if r.AspectRatio != "1:1" {
		t.Errorf("AspectRatio default: got %q", r.AspectRatio)
	}
	if r.Quality != "auto" {
		t.Errorf("Quality default: got %q", r.Quality)
	}
	if r.OutputFormat != "webp" {
		t.Errorf("OutputFormat default: got %q", r.OutputFormat)
	}
	if r.PollInterval == 0 {
		t.Errorf("PollInterval default: got 0")
	}
}

func TestExtractOutputURLs(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{"single string", `"https://x/y.webp"`, []string{"https://x/y.webp"}},
		{"array of strings", `["a","b"]`, []string{"a", "b"}},
		{"mixed any array", `["a", 5, "b"]`, []string{"a", "b"}},
		{"null", `null`, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := extractOutputURLs(json.RawMessage(tc.raw))
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("idx %d: got %q want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestIsTerminalStatus(t *testing.T) {
	for _, tc := range []struct {
		s    string
		term bool
	}{
		{"starting", false},
		{"processing", false},
		{"succeeded", true},
		{"failed", true},
		{"canceled", true},
	} {
		if got := isTerminalStatus(tc.s); got != tc.term {
			t.Errorf("%s: got %v want %v", tc.s, got, tc.term)
		}
	}
}

func TestDetectMime(t *testing.T) {
	if mt := detectMime("/x/y.PNG", nil); mt != "image/png" {
		t.Errorf("png: %q", mt)
	}
	if mt := detectMime("/x/y.jpg", nil); mt != "image/jpeg" {
		t.Errorf("jpg: %q", mt)
	}
	if mt := detectMime("/x/y.webp", nil); mt != "image/webp" {
		t.Errorf("webp: %q", mt)
	}
}

// --- OpenAI provider integration test against an httptest stub ---

func TestOpenAIProviderGenerations(t *testing.T) {
	imgBytes := []byte("\x89PNG\r\n\x1a\nfake")
	encoded := base64.StdEncoding.EncodeToString(imgBytes)

	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"b64_json": encoded, "revised_prompt": "a tidier version"},
			},
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		OpenAI:     &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:          ProviderOpenAI,
		Token:             "test-key",
		Prompt:            "a watercolor fox",
		AspectRatio:       "16:9",
		NumImages:         1,
		OutputFormat:      "webp",
		OutputCompression: 90,
		User:              "tester",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Images) != 1 {
		t.Fatalf("want 1 image, got %d", len(res.Images))
	}
	if string(res.Images[0].Bytes) != string(imgBytes) {
		t.Fatalf("image bytes mismatch")
	}
	if res.Images[0].RevisedPrompt != "a tidier version" {
		t.Errorf("revised_prompt: %q", res.Images[0].RevisedPrompt)
	}

	// Inspect request body
	if captured["model"] != DefaultOpenAIModel {
		t.Errorf("model in body: %v", captured["model"])
	}
	if captured["size"] != "2048x1152" {
		t.Errorf("size in body: %v", captured["size"])
	}
	if captured["output_format"] != "webp" {
		t.Errorf("output_format: %v", captured["output_format"])
	}
	if v, _ := captured["output_compression"].(float64); int(v) != 90 {
		t.Errorf("output_compression: %v", captured["output_compression"])
	}
	if captured["user"] != "tester" {
		t.Errorf("user: %v", captured["user"])
	}
}

func TestOpenAIProviderEditsMultipart(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "ref.png")
	if err := os.WriteFile(imgPath, []byte("fakepng"), 0o644); err != nil {
		t.Fatal(err)
	}
	imgBytes := []byte("output-image-bytes")

	var sawImage bool
	var sawPrompt bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/edits" {
			t.Errorf("path: %s", r.URL.Path)
		}
		ct := r.Header.Get("Content-Type")
		_, params, err := mime.ParseMediaType(ct)
		if err != nil {
			t.Fatalf("ct: %v", err)
		}
		mr := multipart.NewReader(r.Body, params["boundary"])
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			switch part.FormName() {
			case "image[]":
				sawImage = true
			case "prompt":
				b, _ := io.ReadAll(part)
				if string(b) == "compose them" {
					sawPrompt = true
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"b64_json": base64.StdEncoding.EncodeToString(imgBytes)},
			},
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		OpenAI:     &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:    ProviderOpenAI,
		Token:       "tk",
		Prompt:      "compose them",
		InputImages: []string{imgPath},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !sawImage {
		t.Error("server did not see image[] part")
	}
	if !sawPrompt {
		t.Error("server did not see prompt field")
	}
	if string(res.Images[0].Bytes) != string(imgBytes) {
		t.Fatal("output bytes mismatch")
	}
}

// --- Replicate provider integration test ---

func TestReplicateProviderHappyPath(t *testing.T) {
	imgBody := []byte("rendered-webp-bytes")

	var imgServer *httptest.Server
	imgServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/webp")
		_, _ = w.Write(imgBody)
	}))
	defer imgServer.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/models/"):
			if got := r.Header.Get("Authorization"); got != "Bearer rtok" {
				t.Errorf("auth: %q", got)
			}
			body, _ := io.ReadAll(r.Body)
			var b map[string]any
			_ = json.Unmarshal(body, &b)
			input, _ := b["input"].(map[string]any)
			if input["prompt"] != "a fox" {
				t.Errorf("prompt: %v", input["prompt"])
			}
			if input["aspect_ratio"] != "3:2" {
				t.Errorf("aspect_ratio: %v", input["aspect_ratio"])
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "abc",
				"status": "succeeded",
				"output": []string{imgServer.URL + "/out.webp"},
				"urls":   map[string]string{"get": ""},
			})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Prompt:       "a fox",
		AspectRatio:  "3:2",
		NumImages:    1,
		OutputFormat: "webp",
		PollInterval: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Images) != 1 {
		t.Fatalf("want 1 image, got %d", len(res.Images))
	}
	if string(res.Images[0].Bytes) != string(imgBody) {
		t.Fatalf("output bytes mismatch")
	}
}

func TestReplicateProviderPolls(t *testing.T) {
	imgServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("img"))
	}))
	defer imgServer.Close()

	var srv *httptest.Server
	calls := 0
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost:
			// initial create returns processing + a get URL pointing back at us
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "p1",
				"status": "processing",
				"urls":   map[string]string{"get": srv.URL + "/predictions/p1"},
			})
		case r.Method == http.MethodGet:
			calls++
			if calls < 2 {
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "processing", "urls": map[string]string{"get": srv.URL + "/predictions/p1"}})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "succeeded",
				"output": []string{imgServer.URL + "/out"},
			})
		}
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "tk",
		Prompt:       "x",
		AspectRatio:  "1:1",
		NumImages:    1,
		OutputFormat: "webp",
		PollInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Images[0].Bytes) != "img" {
		t.Fatal("bytes")
	}
}

func TestReplicateProviderFailedStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "failed",
			"error":  "rate limited",
		})
	}))
	defer srv.Close()

	c := &Client{
		Replicate: &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	_, err := c.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "tk",
		Prompt:       "x",
		AspectRatio:  "1:1",
		NumImages:    1,
		OutputFormat: "webp",
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("error %q does not contain 'rate limited'", err.Error())
	}
}

func TestIsReplicateHost(t *testing.T) {
	cases := []struct {
		url  string
		want bool
	}{
		{"https://api.replicate.com/v1/predictions/abc", true},
		{"https://pbxt.replicate.delivery/abc.webp", true},
		{"https://replicate.delivery/abc", true},
		{"https://replicate.com/x", true},
		// Token-leak guards: substring match must NOT trigger.
		{"https://evil.example/?x=replicate.delivery", false},
		{"https://api.replicate.com.evil.example/", false},
		{"https://example.com/replicate.com", false},
		{"not a url at all", false},
		{"", false},
	}
	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			if got := isReplicateHost(tc.url); got != tc.want {
				t.Fatalf("isReplicateHost(%q) = %v want %v", tc.url, got, tc.want)
			}
		})
	}
}

func TestRedactRequestBody(t *testing.T) {
	body := map[string]any{
		"input": map[string]any{
			"prompt":         "hi",
			"openai_api_key": "sk-secret-do-not-leak",
		},
	}
	out := redactRequestBody(body)
	if strings.Contains(out, "sk-secret-do-not-leak") {
		t.Fatalf("openai_api_key leaked in log: %s", out)
	}
	if !strings.Contains(out, "REDACTED") {
		t.Fatalf("expected REDACTED placeholder, got: %s", out)
	}
	if !strings.Contains(out, `"prompt":"hi"`) {
		t.Fatalf("expected non-secret fields preserved, got: %s", out)
	}
	// The original body must not be mutated.
	input := body["input"].(map[string]any)
	if input["openai_api_key"] != "sk-secret-do-not-leak" {
		t.Fatalf("redact mutated source map")
	}
}

func TestValidateUnknownAspectRatio(t *testing.T) {
	r := Request{
		Provider:     ProviderOpenAI,
		Token:        "x",
		Prompt:       "x",
		NumImages:    1,
		OutputFormat: "webp",
		AspectRatio:  "weird",
	}
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "unknown aspect ratio") {
		t.Fatalf("expected unknown aspect ratio error, got %v", err)
	}
	// Explicit -size should override and let it pass.
	r.Size = "2048x1152"
	if err := r.Validate(); err != nil {
		t.Fatalf("explicit Size should bypass aspect ratio check: %v", err)
	}
}

func TestValidateEmptyVersion(t *testing.T) {
	r := Request{
		Provider:     ProviderReplicate,
		Token:        "x",
		Model:        "owner/name:",
		Prompt:       "x",
		AspectRatio:  "1:1",
		NumImages:    1,
		OutputFormat: "webp",
	}
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "empty version") {
		t.Fatalf("expected empty version error, got %v", err)
	}
}

func TestParseSize(t *testing.T) {
	cases := []struct {
		in   string
		w, h int
		ok   bool
	}{
		{"1024x1024", 1024, 1024, true},
		{"1920x1080", 1920, 1080, true},
		{"  2048 x 1152 ", 2048, 1152, true},
		{"1024X1024", 1024, 1024, true},
		{"abc", 0, 0, false},
		{"1024", 0, 0, false},
		{"-1x1024", 0, 0, false},
		{"0x100", 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			w, h, ok := ParseSize(tc.in)
			if ok != tc.ok || w != tc.w || h != tc.h {
				t.Fatalf("ParseSize(%q) = %d,%d,%v want %d,%d,%v", tc.in, w, h, ok, tc.w, tc.h, tc.ok)
			}
		})
	}
}

func TestRoundSize(t *testing.T) {
	cases := []struct {
		name         string
		inW, inH     int
		wantW, wantH int
		wantChanged  bool
	}{
		{"1080p+ already aligned", 2048, 1152, 2048, 1152, false},
		{"true 1080p rounds up to 1088", 1920, 1080, 1920, 1088, true},
		{"odd values nearest multiple", 1500, 1000, 1504, 1008, true},
		// 5000→3840 (max-edge), then 3840/1008 > 3:1 forces W down to 3024.
		{"clamp max edge then ratio", 5000, 1000, 3024, 1008, true},
		// 3008/496 > 3:1 forces W down to 1488.
		{"clamp 3:1 ratio", 3000, 500, 1488, 496, true},
		{"floor at edge multiple", 0, 0, 16, 16, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, h, changed := RoundSize(tc.inW, tc.inH)
			if w != tc.wantW || h != tc.wantH || changed != tc.wantChanged {
				t.Fatalf("RoundSize(%d,%d) = %d,%d,%v; want %d,%d,%v",
					tc.inW, tc.inH, w, h, changed, tc.wantW, tc.wantH, tc.wantChanged)
			}
			if w%EdgeMultiple != 0 || h%EdgeMultiple != 0 {
				t.Fatalf("dims must be multiples of %d, got %dx%d", EdgeMultiple, w, h)
			}
		})
	}
}

func TestResolveSizeRoundsCustomSize(t *testing.T) {
	if got := ResolveSize(&Request{Size: "1920x1080"}); got != "1920x1088" {
		t.Fatalf("expected 1920x1088, got %q", got)
	}
	if got := ResolveSize(&Request{Size: "AUTO"}); got != "auto" {
		t.Fatalf("auto should pass through, got %q", got)
	}
	if got := ResolveSize(&Request{AspectRatio: "16:9"}); got != "2048x1152" {
		t.Fatalf("expected 2048x1152, got %q", got)
	}
}

func TestEncodeInputImagesAsDataURLs(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "x.png")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := encodeInputImagesAsDataURLs([]string{
		p,
		"https://example.com/y.jpg",
		"data:image/png;base64,xxx",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out[0], "data:image/png;base64,") {
		t.Errorf("file not encoded: %s", out[0])
	}
	if out[1] != "https://example.com/y.jpg" {
		t.Errorf("url passthrough broken: %s", out[1])
	}
	if out[2] != "data:image/png;base64,xxx" {
		t.Errorf("data url passthrough broken")
	}
}
