package main

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gersham/curds"
)

// pathRecordingStub stands in for Replicate and records every prediction path
// and input, per asset kind, so a test can prove which model a default
// resolved to.
type recordedPrediction struct {
	path  string
	input map[string]any
}

func recordingReplicateStub(t *testing.T, assets map[string]string) (*curds.ReplicateProvider, *[]recordedPrediction) {
	t.Helper()
	assetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/")
		body, ok := assets[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(assetSrv.Close)

	got := &[]recordedPrediction{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(raw, &b)
		input, _ := b["input"].(map[string]any)
		*got = append(*got, recordedPrediction{path: r.URL.Path, input: input})

		out := ""
		switch {
		case strings.Contains(r.URL.Path, "seedance"):
			out = assetSrv.URL + "/out.mp4"
		case strings.Contains(r.URL.Path, "p-image-upscale"):
			out = assetSrv.URL + "/out.png"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "pred-1", "status": "succeeded", "output": out,
		})
	}))
	t.Cleanup(srv.Close)
	return &curds.ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}, got
}

// The September 2026 defaults are what an unqualified run has to reach:
// mp4 → Seedance 2.5, -model tts → Gemini 3.1 Flash TTS, -model upscale →
// prunaai/p-image-upscale.
func TestDefaultModelResolution(t *testing.T) {
	video := "rendered-mp4"
	png := "rendered-png"

	t.Run("mp4 with no -model uses Seedance 2.5", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		rep, got := recordingReplicateStub(t, map[string]string{"out.mp4": video})
		withClient(t, &curds.Client{Replicate: rep})

		out := filepath.Join(t.TempDir(), "clip.mp4")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-prompt", "a slow pan over a desert plain", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*got) != 1 {
			t.Fatalf("predictions: %d", len(*got))
		}
		if (*got)[0].path != "/models/bytedance/seedance-2.5/predictions" {
			t.Errorf("default video model path: %q", (*got)[0].path)
		}
		if data, err := os.ReadFile(out); err != nil || string(data) != video {
			t.Errorf("saved video: %q err=%v", data, err)
		}
	})

	t.Run("-model upscale uses p-image-upscale", func(t *testing.T) {
		ttsTestEnv(t)
		t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
		rep, got := recordingReplicateStub(t, map[string]string{"out.png": png})
		withClient(t, &curds.Client{Replicate: rep})

		src := filepath.Join(t.TempDir(), "small.png")
		if err := os.WriteFile(src, []byte("png-bytes"), 0o644); err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(t.TempDir(), "big.png")
		if _, err := runRealMain(t, []string{
			"-no-tui", "-model", "upscale", "-input-image", src, "-scale", "2", "-output", out,
		}); err != nil {
			t.Fatalf("realMain: %v", err)
		}
		if len(*got) != 1 {
			t.Fatalf("predictions: %d", len(*got))
		}
		if (*got)[0].path != "/models/prunaai/p-image-upscale/predictions" {
			t.Errorf("upscale model path: %q", (*got)[0].path)
		}
		in := (*got)[0].input
		if in["upscale_mode"] != "factor" || in["factor"] != float64(2) || in["output_format"] != "png" {
			t.Errorf("upscale input: %#v", in)
		}
		if data, err := os.ReadFile(out); err != nil || string(data) != png {
			t.Errorf("saved image: %q err=%v", data, err)
		}
	})
}

// Seedance 2.5's dropped resolutions and prunaai's missing face enhancement
// are flag mistakes, so the CLI reports them as usage errors (exit 2) before
// any network call.
func TestFrontierDefaultUsageErrors(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"1080p on seedance 2.5", []string{"-no-tui", "-model", "seedance-2.5", "-prompt", "hi", "-video-resolution", "1080p"}, "-model seedance-2"},
		{"4k on seedance 2.5", []string{"-no-tui", "-model", "seedance-2.5", "-prompt", "hi", "-video-resolution", "4k"}, "-model kling-v3"},
		{"face-enhance on the default upscaler", []string{"-no-tui", "-model", "upscale", "-face-enhance", "-input-image", "x.png"}, "-face-enhance is not supported by -model upscale"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ttsTestEnv(t)
			t.Setenv("REPLICATE_API_TOKEN", "rtok-env")
			_, err := runRealMain(t, tc.args)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want substring %q", err, tc.want)
			}
			var ue *usageError
			if !errors.As(err, &ue) {
				t.Errorf("frontier-default flag problems must exit 2, got %T", err)
			}
		})
	}
}
