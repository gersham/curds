package curds

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
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
		name string
		req  Request
		want string
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
		name       string
		mut        func(r *Request)
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
		{"grok accepts image-to-video", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "xai/grok-imagine-video-1.5"
			r.AspectRatio = "auto"
			r.OutputFormat = "mp4"
			r.VideoDuration = 1
			r.VideoResolution = "480p"
			r.InputImages = []string{"https://example.com/input.png"}
		}, ""},
		{"grok requires one input image", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "xai/grok-imagine-video-1.5"
			r.AspectRatio = "auto"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "720p"
		}, "requires exactly one -input-image"},
		{"grok rejects seedance duration sentinel", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "xai/grok-imagine-video-1.5"
			r.AspectRatio = "auto"
			r.OutputFormat = "mp4"
			r.VideoDuration = -1
			r.VideoResolution = "720p"
			r.InputImages = []string{"https://example.com/input.png"}
		}, "1-15"},
		{"grok rejects 1080p", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "xai/grok-imagine-video-1.5"
			r.AspectRatio = "auto"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "1080p"
			r.InputImages = []string{"https://example.com/input.png"}
		}, "480p or 720p"},
		{"grok rejects seedance references", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "xai/grok-imagine-video-1.5"
			r.AspectRatio = "auto"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "720p"
			r.InputImages = []string{"https://example.com/input.png"}
			r.ReferenceImages = []string{"https://example.com/ref.png"}
		}, "does not support -reference-image"},
		{"seedance accepts video aspect ratio", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "bytedance/seedance-2.0"
			r.AspectRatio = "16:9"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "720p"
		}, ""},
		{"seedance rejects bad duration", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "bytedance/seedance-2.0"
			r.AspectRatio = "16:9"
			r.OutputFormat = "mp4"
			r.VideoDuration = 3
			r.VideoResolution = "720p"
		}, "video_duration"},
		{"seedance accepts multiple input images as references", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "bytedance/seedance-2.0"
			r.AspectRatio = "16:9"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "720p"
			r.InputImages = []string{"https://example.com/1.png", "https://example.com/2.png"}
		}, ""},
		{"seedance last frame requires exactly one input image", func(r *Request) {
			r.Provider = ProviderReplicate
			r.Model = "bytedance/seedance-2.0"
			r.AspectRatio = "16:9"
			r.OutputFormat = "mp4"
			r.VideoDuration = 5
			r.VideoResolution = "720p"
			r.InputImages = []string{"https://example.com/1.png", "https://example.com/2.png"}
			r.LastFrameImage = "https://example.com/end.png"
		}, "exactly one"},
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

func TestOpenAIProviderEditsConvertsJPEGToPNG(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "photo.jpeg")
	src := image.NewRGBA(image.Rect(0, 0, 8, 8))
	var jbuf bytes.Buffer
	if err := jpeg.Encode(&jbuf, src, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(imgPath, jbuf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	var partName, partType string
	var partBytes []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
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
			if part.FormName() == "image[]" {
				partName = part.FileName()
				partType = part.Header.Get("Content-Type")
				partBytes, _ = io.ReadAll(part)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"b64_json": base64.StdEncoding.EncodeToString([]byte("out"))},
			},
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		OpenAI:     &OpenAIProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	if _, err := c.Generate(context.Background(), &Request{
		Provider:    ProviderOpenAI,
		Token:       "tk",
		Prompt:      "edit it",
		InputImages: []string{imgPath},
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if partName != "photo.png" {
		t.Errorf("filename: %q, want photo.png", partName)
	}
	if partType != "image/png" {
		t.Errorf("content-type: %q, want image/png", partType)
	}
	if _, err := png.Decode(bytes.NewReader(partBytes)); err != nil {
		t.Errorf("uploaded bytes are not valid PNG: %v", err)
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

func TestReplicateProviderSeedanceVideoHappyPath(t *testing.T) {
	videoBody := []byte("rendered-mp4-bytes")

	videoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write(videoBody)
	}))
	defer videoServer.Close()

	audio := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasPrefix(r.URL.Path, "/models/bytedance/seedance-2.0/predictions") {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(body, &b)
		input, _ := b["input"].(map[string]any)
		if input["prompt"] != "a glass sculpture forming" {
			t.Errorf("prompt: %v", input["prompt"])
		}
		if input["duration"] != float64(5) {
			t.Errorf("duration: %v", input["duration"])
		}
		if input["resolution"] != "720p" {
			t.Errorf("resolution: %v", input["resolution"])
		}
		if input["aspect_ratio"] != "16:9" {
			t.Errorf("aspect_ratio: %v", input["aspect_ratio"])
		}
		if input["generate_audio"] != false {
			t.Errorf("generate_audio: %v", input["generate_audio"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "vid1",
			"status": "succeeded",
			"output": videoServer.URL + "/out.mp4",
			"urls":   map[string]string{"get": ""},
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:        ProviderReplicate,
		Token:           "rtok",
		Model:           "bytedance/seedance-2.0",
		Prompt:          "a glass sculpture forming",
		AspectRatio:     "16:9",
		OutputFormat:    "mp4",
		VideoDuration:   5,
		VideoResolution: "720p",
		GenerateAudio:   &audio,
		PollInterval:    10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Videos) != 1 {
		t.Fatalf("want 1 video, got %d", len(res.Videos))
	}
	if string(res.Videos[0].Bytes) != string(videoBody) {
		t.Fatalf("video bytes mismatch")
	}
}

func TestReplicateProviderGrokImagineVideoHappyPath(t *testing.T) {
	videoBody := []byte("rendered-grok-mp4-bytes")

	videoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write(videoBody)
	}))
	defer videoServer.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasPrefix(r.URL.Path, "/models/xai/grok-imagine-video-1.5/predictions") {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(body, &b)
		input, _ := b["input"].(map[string]any)
		if input["prompt"] != "animate the product photo" {
			t.Errorf("prompt: %v", input["prompt"])
		}
		if input["image"] != "https://example.com/product.png" {
			t.Errorf("image: %v", input["image"])
		}
		if input["duration"] != float64(5) {
			t.Errorf("duration: %v", input["duration"])
		}
		if input["resolution"] != "720p" {
			t.Errorf("resolution: %v", input["resolution"])
		}
		if input["aspect_ratio"] != "auto" {
			t.Errorf("aspect_ratio: %v", input["aspect_ratio"])
		}
		if _, ok := input["generate_audio"]; ok {
			t.Errorf("generate_audio should not be sent to Grok Imagine Video")
		}
		if _, ok := input["reference_images"]; ok {
			t.Errorf("reference_images should not be sent to Grok Imagine Video")
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "grok-vid1",
			"status": "succeeded",
			"output": videoServer.URL + "/out.mp4",
			"urls":   map[string]string{"get": ""},
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:        ProviderReplicate,
		Token:           "rtok",
		Model:           "xai/grok-imagine-video-1.5",
		Prompt:          "animate the product photo",
		AspectRatio:     "auto",
		OutputFormat:    "mp4",
		VideoDuration:   5,
		VideoResolution: "720p",
		InputImages:     []string{"https://example.com/product.png"},
		PollInterval:    10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Videos) != 1 {
		t.Fatalf("want 1 video, got %d", len(res.Videos))
	}
	if string(res.Videos[0].Bytes) != string(videoBody) {
		t.Fatalf("video bytes mismatch")
	}
}

func TestReplicateProviderSeedanceInputImages(t *testing.T) {
	videoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("video"))
	}))
	defer videoServer.Close()

	cases := []struct {
		name             string
		inputImages      []string
		wantImage        string
		wantReferenceLen int
	}{
		{
			name:        "single input image is first frame",
			inputImages: []string{"https://example.com/first.png"},
			wantImage:   "https://example.com/first.png",
		},
		{
			name:             "multiple input images are references",
			inputImages:      []string{"https://example.com/one.png", "https://example.com/two.png"},
			wantReferenceLen: 2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var b map[string]any
				_ = json.Unmarshal(body, &b)
				input, _ := b["input"].(map[string]any)
				if got, _ := input["image"].(string); got != tc.wantImage {
					t.Errorf("image: %q", got)
				}
				refs, _ := input["reference_images"].([]any)
				if len(refs) != tc.wantReferenceLen {
					t.Errorf("reference_images len: %d", len(refs))
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id":     "vid-input",
					"status": "succeeded",
					"output": videoServer.URL + "/out.mp4",
				})
			}))
			defer srv.Close()

			c := &Client{
				HTTPClient: srv.Client(),
				Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
			}
			_, err := c.Generate(context.Background(), &Request{
				Provider:        ProviderReplicate,
				Token:           "rtok",
				Model:           "bytedance/seedance-2.0",
				Prompt:          "animate the image",
				AspectRatio:     "16:9",
				OutputFormat:    "mp4",
				VideoDuration:   5,
				VideoResolution: "720p",
				InputImages:     tc.inputImages,
				PollInterval:    10 * time.Millisecond,
			})
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
		})
	}
}

func TestIsVideoModel(t *testing.T) {
	cases := map[string]bool{
		"xai/grok-imagine-video-1.5":          true,
		"XAI/Grok-Imagine-Video-1.5":          true,
		"xai/grok-imagine-video-1.5:abc123":   true,
		"bytedance/seedance-2.0":              true,
		"bytedance/seedance-2.0-fast":         true,
		"bytedance/seedance-2.0:abc123":       true,
		"openai/gpt-image-2":                  false,
		"xai/grok-imagine-video-1.5-evil/foo": false,
		"grok-imagine-video":                  true,
		"Grok-Imagine-Video":                  true,
		"":                                    false,
	}
	for in, want := range cases {
		if got := IsVideoModel(in); got != want {
			t.Errorf("IsVideoModel(%q) = %v want %v", in, got, want)
		}
	}
}

func TestIsXaiVideoModel(t *testing.T) {
	cases := map[string]bool{
		"grok-imagine-video":         true,
		"Grok-Imagine-Video":         true,
		"  grok-imagine-video  ":     true,
		"xai/grok-imagine-video-1.5": false,
		"grok-imagine-video-1.5":     false,
		"":                           false,
	}
	for in, want := range cases {
		if got := IsXaiVideoModel(in); got != want {
			t.Errorf("IsXaiVideoModel(%q) = %v want %v", in, got, want)
		}
	}
}

func TestRequestValidateXaiVideo(t *testing.T) {
	base := func() Request {
		return Request{
			Provider:        ProviderXai,
			Token:           "xk",
			Model:           "grok-imagine-video",
			Prompt:          "a serene time-lapse",
			NumImages:       1,
			OutputFormat:    "mp4",
			AspectRatio:     "auto",
			VideoResolution: "720p",
		}
	}

	t.Run("text-to-video ok (no image)", func(t *testing.T) {
		r := base()
		if err := r.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("image-to-video ok", func(t *testing.T) {
		r := base()
		r.InputImages = []string{"https://example.com/a.png"}
		if err := r.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("reference image ok", func(t *testing.T) {
		r := base()
		r.InputImages = []string{"https://example.com/a.png"}
		r.ReferenceImages = []string{"https://example.com/ref1.png", "https://example.com/ref2.png"}
		if err := r.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("1080p ok", func(t *testing.T) {
		r := base()
		r.VideoResolution = "1080p"
		if err := r.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("wrong provider rejected", func(t *testing.T) {
		r := base()
		r.Provider = ProviderReplicate
		if err := r.Validate(); err == nil {
			t.Fatal("expected provider error")
		}
	})
	t.Run("too many input images rejected", func(t *testing.T) {
		r := base()
		r.InputImages = []string{"a.png", "b.png"}
		if err := r.Validate(); err == nil {
			t.Fatal("expected input-image count error")
		}
	})
	t.Run("bad duration rejected", func(t *testing.T) {
		r := base()
		r.VideoDuration = 20
		if err := r.Validate(); err == nil {
			t.Fatal("expected duration error")
		}
	})
	t.Run("no-audio rejected", func(t *testing.T) {
		r := base()
		no := false
		r.GenerateAudio = &no
		if err := r.Validate(); err == nil {
			t.Fatal("expected no-audio error")
		}
	})
	t.Run("seed rejected", func(t *testing.T) {
		r := base()
		r.Seed = 7
		if err := r.Validate(); err == nil {
			t.Fatal("expected seed error")
		}
	})
}

func TestXaiProviderHappyPath(t *testing.T) {
	videoBody := []byte("rendered-xai-mp4-bytes")

	videoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write(videoBody)
	}))
	defer videoServer.Close()

	var srv *httptest.Server
	polls := 0
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/videos/generations":
			body, _ := io.ReadAll(r.Body)
			var b map[string]any
			_ = json.Unmarshal(body, &b)
			if b["model"] != "grok-imagine-video" {
				t.Errorf("model: %v", b["model"])
			}
			if b["prompt"] != "animate the product photo" {
				t.Errorf("prompt: %v", b["prompt"])
			}
			if b["duration"] != float64(10) {
				t.Errorf("duration: %v", b["duration"])
			}
			if b["resolution"] != "1080p" {
				t.Errorf("resolution: %v", b["resolution"])
			}
			// aspect_ratio "auto" must be omitted.
			if _, ok := b["aspect_ratio"]; ok {
				t.Errorf("aspect_ratio should be omitted for auto, got %v", b["aspect_ratio"])
			}
			img, ok := b["image"].(map[string]any)
			if !ok || img["url"] != "https://example.com/product.png" {
				t.Errorf("image: %v", b["image"])
			}
			refs, ok := b["reference_images"].([]any)
			if !ok || len(refs) != 1 {
				t.Errorf("reference_images: %v", b["reference_images"])
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"request_id": "req-123"})
		case r.Method == http.MethodGet && r.URL.Path == "/videos/req-123":
			polls++
			w.Header().Set("Content-Type", "application/json")
			if polls < 2 {
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "pending", "progress": 40})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status":   "done",
				"progress": 100,
				"video":    map[string]any{"url": videoServer.URL + "/out.mp4", "duration": 10},
			})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Xai:        &XaiProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:        ProviderXai,
		Token:           "xk",
		Model:           "grok-imagine-video",
		Prompt:          "animate the product photo",
		AspectRatio:     "auto",
		OutputFormat:    "mp4",
		VideoDuration:   10,
		VideoResolution: "1080p",
		InputImages:     []string{"https://example.com/product.png"},
		ReferenceImages: []string{"https://example.com/ref.png"},
		PollInterval:    5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Videos) != 1 {
		t.Fatalf("want 1 video, got %d", len(res.Videos))
	}
	if string(res.Videos[0].Bytes) != string(videoBody) {
		t.Fatalf("video bytes mismatch")
	}
	if polls < 2 {
		t.Fatalf("expected polling, got %d polls", polls)
	}
}

func TestXaiProviderFailedStatus(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_ = json.NewEncoder(w).Encode(map[string]any{"request_id": "req-x"})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "failed",
			"error":  map[string]any{"code": "invalid_argument", "message": "bad prompt"},
		})
	}))
	defer srv.Close()

	c := &Client{Xai: &XaiProvider{HTTPClient: srv.Client(), APIBase: srv.URL}}
	_, err := c.Generate(context.Background(), &Request{
		Provider:        ProviderXai,
		Token:           "xk",
		Model:           "grok-imagine-video",
		Prompt:          "x",
		OutputFormat:    "mp4",
		AspectRatio:     "auto",
		VideoResolution: "720p",
		PollInterval:    5 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected failure error")
	}
	if !strings.Contains(err.Error(), "bad prompt") {
		t.Fatalf("error should surface upstream message, got: %v", err)
	}
}

func TestIsXaiHost(t *testing.T) {
	cases := map[string]bool{
		"https://vidgen.x.ai/bucket/v.mp4": true,
		"https://x.ai/v.mp4":               true,
		"https://api.x.ai/v1/videos/1":     true,
		"https://evil.example/?x.ai":       false,
		"https://notx.ai.evil.com/v.mp4":   false,
		"://bad":                           false,
	}
	for in, want := range cases {
		if got := isXaiHost(in); got != want {
			t.Errorf("isXaiHost(%q) = %v want %v", in, got, want)
		}
	}
}

func TestIsSegmentationModel(t *testing.T) {
	cases := map[string]bool{
		"bria/remove-background":          true,
		"BRIA/Remove-Background":          true,
		"bria/remove-background:abc123":   true,
		"openai/gpt-image-2":              false,
		"bytedance/seedance-2.0":          false,
		"":                                false,
		"bria/remove-background-evil/foo": false,
	}
	for in, want := range cases {
		if got := IsSegmentationModel(in); got != want {
			t.Errorf("IsSegmentationModel(%q) = %v want %v", in, got, want)
		}
	}
}

func TestRequestValidateSegmentation(t *testing.T) {
	base := Request{
		Provider:     ProviderReplicate,
		Token:        "tk",
		Model:        "bria/remove-background",
		NumImages:    1,
		OutputFormat: "png",
		InputImages:  []string{"https://example.com/cat.jpg"},
	}
	cases := []struct {
		name       string
		mut        func(r *Request)
		wantErrSub string
	}{
		{"valid no prompt", func(r *Request) {}, ""},
		{"prompt allowed but ignored", func(r *Request) { r.Prompt = "anything" }, ""},
		{"requires input image", func(r *Request) { r.InputImages = nil }, "requires exactly one"},
		{"rejects multiple input images", func(r *Request) {
			r.InputImages = []string{"a", "b"}
		}, "requires exactly one"},
		{"rejects mask", func(r *Request) { r.Mask = "m.png" }, "does not accept -mask"},
		{"rejects size", func(r *Request) { r.Size = "1024x1024" }, "preserves the input"},
		{"rejects non-png output", func(r *Request) { r.OutputFormat = "webp" }, "must be png"},
		{"rejects num_images > 1", func(r *Request) { r.NumImages = 3 }, "produces exactly one"},
		{"rejects openai provider", func(r *Request) {
			r.Provider = ProviderOpenAI
		}, "only supported with provider replicate"},
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
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("want error containing %q, got %v", tc.wantErrSub, err)
			}
		})
	}
}

func TestReplicateProviderSegmentation(t *testing.T) {
	pngBody := []byte("\x89PNGfake-cutout-bytes")

	imgServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(pngBody)
	}))
	defer imgServer.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost ||
			!strings.HasPrefix(r.URL.Path, "/models/bria/remove-background/predictions") {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(body, &b)
		input, _ := b["input"].(map[string]any)
		if _, ok := input["prompt"]; ok {
			t.Errorf("segmentation must NOT send a prompt field, body was: %v", input)
		}
		if got, _ := input["image"].(string); got != "https://example.com/cat.jpg" {
			t.Errorf("image: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "seg1",
			"status": "succeeded",
			"output": imgServer.URL + "/cutout.png",
		})
	}))
	defer srv.Close()

	c := &Client{
		HTTPClient: srv.Client(),
		Replicate:  &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL},
	}
	res, err := c.Generate(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        "bria/remove-background",
		InputImages:  []string{"https://example.com/cat.jpg"},
		OutputFormat: "png",
		PollInterval: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Images) != 1 {
		t.Fatalf("want 1 image, got %d", len(res.Images))
	}
	if string(res.Images[0].Bytes) != string(pngBody) {
		t.Fatalf("output bytes mismatch: %q", res.Images[0].Bytes)
	}
	if res.Images[0].Format != "png" {
		t.Errorf("format: %q", res.Images[0].Format)
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

func TestEncodeInlineImage(t *testing.T) {
	data := []byte("\x89PNGfake")
	out := EncodeInlineImage(data, InlineImageOpts{
		Name:           "x.png",
		WidthCells:     40,
		HeightCells:    20,
		PreserveAspect: true,
	})
	if !strings.HasPrefix(out, "\x1b]1337;File=") {
		t.Fatalf("missing OSC 1337 prefix: %q", out[:30])
	}
	if !strings.Contains(out, "inline=1") {
		t.Errorf("expected inline=1: %q", out)
	}
	if !strings.Contains(out, "width=40") || !strings.Contains(out, "height=20") {
		t.Errorf("expected width/height: %q", out)
	}
	if !strings.Contains(out, "preserveAspectRatio=1") {
		t.Errorf("expected preserveAspectRatio: %q", out)
	}
	if !strings.HasSuffix(out, "\x07") {
		t.Errorf("expected BEL terminator: %q", out[len(out)-2:])
	}
	// Encoded body should be present (base64 of "\x89PNGfake")
	expected := base64.StdEncoding.EncodeToString(data)
	if !strings.Contains(out, expected) {
		t.Errorf("missing base64 body %q in: %q", expected, out)
	}
}

func TestEncodeInlineImageTmuxWrap(t *testing.T) {
	t.Setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
	out := EncodeInlineImage([]byte("data"), InlineImageOpts{Name: "a.png"})
	if !strings.HasPrefix(out, "\x1bPtmux;") {
		t.Fatalf("expected tmux passthrough prefix: %q", out[:20])
	}
	if !strings.HasSuffix(out, "\x1b\\") {
		t.Fatalf("expected ST terminator: %q", out[len(out)-4:])
	}
	// ESCs inside the body must be doubled.
	if strings.Count(out, "\x1b\x1b") < 1 {
		t.Errorf("expected doubled ESC inside tmux envelope: %q", out)
	}
}

func TestSupportsInlineImages(t *testing.T) {
	cases := []struct {
		env  map[string]string
		want bool
	}{
		{map[string]string{"TERM_PROGRAM": "iTerm.app"}, true},
		{map[string]string{"TERM_PROGRAM": "WezTerm"}, true},
		{map[string]string{"TERM_PROGRAM": "vscode"}, true},
		{map[string]string{"TERM_PROGRAM": "Apple_Terminal"}, false},
		{map[string]string{"TERM_PROGRAM": "tmux"}, false},
		{map[string]string{"WEZTERM_EXECUTABLE": "/usr/bin/wezterm"}, true},
		{map[string]string{"KONSOLE_VERSION": "240800"}, true},
		{map[string]string{}, false},
	}
	for i, tc := range cases {
		// Each subtest gets a fresh isolated env so cases don't leak.
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			t.Setenv("TERM_PROGRAM", "")
			t.Setenv("WEZTERM_EXECUTABLE", "")
			t.Setenv("KONSOLE_VERSION", "")
			for k, v := range tc.env {
				t.Setenv(k, v)
			}
			if got := SupportsInlineImages(); got != tc.want {
				t.Fatalf("env=%v: got %v, want %v", tc.env, got, tc.want)
			}
		})
	}
}

func TestEncodeMediaAsDataURLs(t *testing.T) {
	tmp := t.TempDir()
	p := filepath.Join(tmp, "x.png")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := encodeMediaAsDataURLs([]string{
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
