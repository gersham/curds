package main

import (
	"encoding/json"
	"errors"
	"fmt"
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

// withRunProvider points `curds run` at an httptest server for the duration of
// the test.
func withRunProvider(t *testing.T, srv *httptest.Server) {
	t.Helper()
	old := newRunProvider
	newRunProvider = func() *curds.ReplicateProvider {
		return &curds.ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	}
	t.Cleanup(func() { newRunProvider = old })
}

// captureStdout runs fn with os.Stdout redirected and returns what it printed.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	fn()
	_ = w.Close()
	out := <-done
	os.Stdout = old
	return out
}

// TestRunSubcommandEndToEnd drives `curds run` against a fake Replicate: it must
// POST the caller's inputs verbatim, poll, download every output, and name the
// files with -1/-2 suffixes before the extension.
func TestRunSubcommandEndToEnd(t *testing.T) {
	assets := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = fmt.Fprintf(w, "asset%s", strings.TrimPrefix(r.URL.Path, "/"))
	}))
	defer assets.Close()

	var gotInput map[string]any
	var gotPath, gotAuth string
	gets := 0
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			gotPath = r.URL.Path
			gotAuth = r.Header.Get("Authorization")
			body, _ := io.ReadAll(r.Body)
			var b map[string]any
			if err := json.Unmarshal(body, &b); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			gotInput, _ = b["input"].(map[string]any)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "run-1", "status": "starting",
				"urls": map[string]string{"get": srv.URL + "/predictions/run-1"},
			})
		case http.MethodGet:
			gets++
			if gets < 2 {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "run-1", "status": "processing",
					"urls": map[string]string{"get": srv.URL + "/predictions/run-1"},
				})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "run-1", "status": "succeeded",
				"output": []string{assets.URL + "/one.mp4", assets.URL + "/two.mp4"},
			})
		}
	}))
	defer srv.Close()
	withRunProvider(t, srv)
	t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))

	out := filepath.Join(t.TempDir(), "clip.mp4")
	ref := filepath.Join(t.TempDir(), "ref.png")
	if err := os.WriteFile(ref, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}

	var runErr error
	printed := captureStdout(t, func() {
		runErr = realMainRun(newLogger(io.Discard), time.Now(), []string{
			"-output", out, "-token", "rtok", "-poll-interval", "1ms",
			"owner/model", "prompt=a slow pan", "duration=5", "ref=@" + ref,
		})
	})
	if runErr != nil {
		t.Fatalf("realMainRun: %v", runErr)
	}

	if gotPath != "/models/owner/model/predictions" {
		t.Errorf("path: %q", gotPath)
	}
	if gotAuth != "Bearer rtok" {
		t.Errorf("authorization: %q", gotAuth)
	}
	if gotInput["prompt"] != "a slow pan" || gotInput["duration"] != float64(5) {
		t.Errorf("inputs: %#v", gotInput)
	}
	if s, _ := gotInput["ref"].(string); !strings.HasPrefix(s, "data:image/png;base64,") {
		t.Errorf("@file should upload: %#v", gotInput["ref"])
	}
	if gets < 2 {
		t.Errorf("expected polling, got %d GETs", gets)
	}

	// Multi-output naming: -1, -2 before the extension.
	want := []string{
		strings.TrimSuffix(out, ".mp4") + "-1.mp4",
		strings.TrimSuffix(out, ".mp4") + "-2.mp4",
	}
	for i, p := range want {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("output %d missing: %v", i, err)
		}
		if string(b) != []string{"assetone.mp4", "assettwo.mp4"}[i] {
			t.Errorf("output %d bytes: %q", i, b)
		}
	}
	for _, p := range want {
		if !strings.Contains(printed, p) {
			t.Errorf("stdout should print %s, got %q", p, printed)
		}
	}
}

// -json prints the prediction JSON instead of downloading anything.
func TestRunSubcommandJSONOutput(t *testing.T) {
	assets := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("no download expected with -json")
	}))
	defer assets.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "run-2", "status": "succeeded", "output": assets.URL + "/one.mp4",
		})
	}))
	defer srv.Close()
	withRunProvider(t, srv)
	t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))

	out := filepath.Join(t.TempDir(), "ignored.mp4")
	var runErr error
	printed := captureStdout(t, func() {
		runErr = realMainRun(newLogger(io.Discard), time.Now(), []string{
			"-json", "-output", out, "-token", "rtok", "owner/model", "prompt=hi",
		})
	})
	if runErr != nil {
		t.Fatalf("realMainRun: %v", runErr)
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("-json must not download")
	}
	var prediction map[string]any
	if err := json.Unmarshal([]byte(printed), &prediction); err != nil {
		t.Fatalf("stdout is not the prediction JSON: %v (%q)", err, printed)
	}
	if prediction["status"] != "succeeded" {
		t.Errorf("prediction: %#v", prediction)
	}
}

// Outputs with no downloadable URLs — objects, plain strings, streamed token
// arrays from text models — print the output JSON instead of downloading.
func TestRunSubcommandNonURLOutputPrintsJSON(t *testing.T) {
	cases := []struct {
		name   string
		output any
		want   string
	}{
		{"object", map[string]any{"text": "hi there"}, "hi there"},
		{"plain string", "hello world", "hello world"},
		{"token array", []string{"hel", "lo"}, `"lo"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet && !strings.HasPrefix(r.URL.Path, "/predictions") {
					t.Errorf("unexpected download of %s", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "run-3", "status": "succeeded", "output": tc.output,
				})
			}))
			defer srv.Close()
			withRunProvider(t, srv)
			t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))

			var runErr error
			printed := captureStdout(t, func() {
				runErr = realMainRun(newLogger(io.Discard), time.Now(), []string{
					"-token", "rtok", "owner/text-model", "prompt=hi",
				})
			})
			if runErr != nil {
				t.Fatalf("realMainRun: %v", runErr)
			}
			if !strings.Contains(printed, tc.want) {
				t.Errorf("stdout should carry the output JSON, got %q", printed)
			}
		})
	}
}

// -schema prints one line per input and creates no prediction.
func TestRunSubcommandSchema(t *testing.T) {
	posted := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posted = true
		}
		if r.URL.Path != "/models/sync/lipsync-2-pro" {
			t.Errorf("path: %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"latest_version":{"openapi_schema":{"components":{"schemas":{
			"Input":{"properties":{
				"video":{"type":"string","description":"Source video"},
				"sync_mode":{"allOf":[{"$ref":"#/components/schemas/sync_mode"}]}
			}},
			"sync_mode":{"type":"string","enum":["loop","bounce"],"default":"loop"}
		}}}}}`))
	}))
	defer srv.Close()
	withRunProvider(t, srv)
	t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))

	var runErr error
	printed := captureStdout(t, func() {
		runErr = realMainRun(newLogger(io.Discard), time.Now(), []string{
			"-schema", "-token", "rtok", "sync/lipsync-2-pro",
		})
	})
	if runErr != nil {
		t.Fatalf("realMainRun: %v", runErr)
	}
	if posted {
		t.Error("-schema must not create a prediction")
	}
	lines := strings.Split(strings.TrimSpace(printed), "\n")
	if len(lines) != 2 {
		t.Fatalf("want one line per input, got %q", printed)
	}
	// Plain alphanumeric values must render unquoted: the logfmt quoter only
	// quotes values that would read back ambiguously.
	if !strings.HasPrefix(lines[0], "name=sync_mode") {
		t.Errorf("sorted output expected, got %q", lines[0])
	}
	if !strings.Contains(lines[0], "enum=[loop,bounce]") {
		t.Errorf("enum from $ref missing: %q", lines[0])
	}
	if !strings.Contains(lines[0], "type=string") || !strings.Contains(lines[0], "default=loop") {
		t.Errorf("plain values must stay unquoted: %q", lines[0])
	}
	if !strings.Contains(lines[1], "name=video") || !strings.Contains(lines[1], "description=\"Source video\"") {
		t.Errorf("description missing: %q", lines[1])
	}
}

func TestRunSubcommandUsageErrors(t *testing.T) {
	t.Setenv("CURDS_CONFIG", filepath.Join(t.TempDir(), "config.toml"))
	logger := newLogger(io.Discard)

	cases := []struct {
		name string
		args []string
		want string
	}{
		{"no model", []string{"-token", "rtok"}, "OWNER/MODEL"},
		{"model must be owner/name", []string{"-token", "rtok", "seedance-2"}, "owner/name"},
		{"bad value", []string{"-token", "rtok", "owner/model", "notapair"}, "key=value"},
		{"schema takes no inputs", []string{"-token", "rtok", "-schema", "owner/model", "prompt=x"}, "no input arguments"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := captureStdoutErr(t, func() error {
				return realMainRun(logger, time.Now(), tc.args)
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want substring %q", err, tc.want)
			}
			var ue *usageError
			if !errors.As(err, &ue) {
				t.Errorf("usage problems must exit 2, got %T", err)
			}
		})
	}
}

// captureStdoutErr keeps a test that is only interested in the error from
// polluting the test log with help text.
func captureStdoutErr(t *testing.T, fn func() error) error {
	t.Helper()
	var err error
	captureStdout(t, func() { err = fn() })
	return err
}

func TestRunOutputExt(t *testing.T) {
	cases := []struct {
		name string
		urls []string
		want string
	}{
		{"mp4", []string{"https://replicate.delivery/out.mp4"}, "mp4"},
		{"first with extension wins", []string{"https://x/opaque", "https://x/out.webm"}, "webm"},
		{"query string stripped", []string{"https://x/out.png?sig=1"}, "png"},
		{"no extension", []string{"https://x/predictions/abc"}, "bin"},
		{"empty", nil, "bin"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := runOutputExt(tc.urls); got != tc.want {
				t.Errorf("runOutputExt(%v) = %q want %q", tc.urls, got, tc.want)
			}
		})
	}
}

func TestValidateMediaInputs(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		opts       *cliOptions
		wantErrSub string
	}{
		{
			name:  "kling-avatar complete",
			model: curds.KlingAvatarModel,
			opts:  &cliOptions{inputImages: imageList{"portrait.png"}, audio: "voice.mp3"},
		},
		{
			name: "kling-avatar without audio", model: curds.KlingAvatarModel,
			opts:       &cliOptions{inputImages: imageList{"portrait.png"}},
			wantErrSub: "requires -audio",
		},
		{
			name: "kling-avatar without portrait", model: curds.KlingAvatarModel,
			opts:       &cliOptions{audio: "voice.mp3"},
			wantErrSub: "exactly one -input-image",
		},
		{
			name:  "lipsync complete",
			model: curds.LipsyncModel,
			opts:  &cliOptions{inputVideo: "clip.mp4", audio: "voice.wav"},
		},
		{
			name: "lipsync without input video", model: curds.LipsyncModel,
			opts:       &cliOptions{audio: "voice.wav"},
			wantErrSub: "requires -input-video",
		},
		{
			name: "lipsync without audio", model: curds.LipsyncModel,
			opts:       &cliOptions{inputVideo: "clip.mp4"},
			wantErrSub: "requires -audio",
		},
		{
			name:  "other models are untouched",
			model: "bytedance/seedance-2.0",
			opts:  &cliOptions{},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMediaInputs(tc.model, tc.opts)
			if tc.wantErrSub == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("got %v, want substring %q", err, tc.wantErrSub)
			}
			// realMain wraps this in a usageError, which exits 2.
			var ue *usageError
			if !errors.As(&usageError{err: err}, &ue) {
				t.Error("media-input problems must be usage errors")
			}
		})
	}
}

// The new models and flags must be discoverable from the help text, and the run
// subcommand must document itself.
func TestHelpTextCoversTalkingHeadsAndRun(t *testing.T) {
	main := helpText()
	for _, want := range []string{
		"curds run", "RUN SUBCOMMAND", "kling-avatar", "lipsync",
		"-audio", "-input-video", "-sync-mode", "-sync-temperature",
		"-active-speaker", "-crop-captions", "-no-fallback",
		"SEEDANCE FACE REJECTION",
	} {
		if !strings.Contains(main, want) {
			t.Errorf("help text is missing %q", want)
		}
	}

	run := runHelpText()
	for _, want := range []string{
		"OWNER/MODEL", "-schema", "-json", "-output", "-poll-interval",
		"REPLICATE_API_TOKEN", "key=value",
	} {
		if !strings.Contains(run, want) {
			t.Errorf("run help text is missing %q", want)
		}
	}
}
