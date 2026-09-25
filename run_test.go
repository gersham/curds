package curds

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseRunInput(t *testing.T) {
	dir := t.TempDir()
	media := filepath.Join(dir, "ref.png")
	if err := os.WriteFile(media, []byte("png-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	refA := filepath.Join(dir, "a.png")
	refB := filepath.Join(dir, "b.png")
	for _, p := range []string{refA, refB} {
		if err := os.WriteFile(p, []byte("ref"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := ParseRunInput([]string{
		"prompt=a slow pan",
		"duration=5",
		"active_speaker=true",
		"ratio=\"16:9\"",
		"seed=null",
		"tags=[\"a\",\"b\"]",
		"meta={\"k\":1}",
		"image=@" + media,
		"remote=@https://example.com/ref.png",
		"empty=",
		"ref=@" + refA,
		"ref=@" + refB,
	})
	if err != nil {
		t.Fatalf("ParseRunInput: %v", err)
	}

	if got["prompt"] != "a slow pan" {
		t.Errorf("prompt: %#v", got["prompt"])
	}
	if got["duration"] != float64(5) {
		t.Errorf("duration should decode as a number: %#v", got["duration"])
	}
	if got["active_speaker"] != true {
		t.Errorf("active_speaker should decode as a bool: %#v", got["active_speaker"])
	}
	if got["ratio"] != "16:9" {
		t.Errorf("quoted JSON string: %#v", got["ratio"])
	}
	if got["seed"] != nil {
		t.Errorf("null: %#v", got["seed"])
	}
	if arr, ok := got["tags"].([]any); !ok || len(arr) != 2 {
		t.Errorf("tags should stay a JSON array: %#v", got["tags"])
	}
	if obj, ok := got["meta"].(map[string]any); !ok || obj["k"] != float64(1) {
		t.Errorf("meta should decode as an object: %#v", got["meta"])
	}
	if s, _ := got["image"].(string); !strings.HasPrefix(s, "data:image/png;base64,") {
		t.Errorf("@file should upload as a data URL: %q", got["image"])
	}
	if got["remote"] != "https://example.com/ref.png" {
		t.Errorf("@url should pass through: %#v", got["remote"])
	}
	if got["empty"] != "" {
		t.Errorf("bare key= should be the empty string: %#v", got["empty"])
	}
	// Repeated keys collect into an array, in the order given.
	refs, ok := got["ref"].([]any)
	if !ok || len(refs) != 2 {
		t.Fatalf("repeated keys should build an array: %#v", got["ref"])
	}
	for i := range refs {
		s, _ := refs[i].(string)
		if !strings.HasPrefix(s, "data:image/png;base64,") {
			t.Errorf("repeated key value %d should be an uploaded data URL: %q", i, s)
		}
	}
	if _, ok := got["ref"].([]string); ok {
		t.Error("repeated keys must not send a []string (JSON-encodes as an array)")
	}
}

func TestParseRunInputErrors(t *testing.T) {
	if _, err := ParseRunInput([]string{"not-a-pair"}); err == nil || !strings.Contains(err.Error(), "key=value") {
		t.Errorf("missing '=': %v", err)
	}
	if _, err := ParseRunInput([]string{"=value"}); err == nil || !strings.Contains(err.Error(), "empty key") {
		t.Errorf("empty key: %v", err)
	}
	if _, err := ParseRunInput([]string{"image=@/nope/missing.png"}); err == nil || !strings.Contains(err.Error(), "image") {
		t.Errorf("unreadable @file: %v", err)
	}
	if _, err := ParseRunInput([]string{"ref=@a.png,@b.png"}); err == nil {
		t.Error("a comma-joined @list is not special: it must fail on the unreadable path, not be split")
	}
}

// Run must create one prediction with exactly the caller's inputs, poll it, and
// hand back the raw output plus any URLs it carried.
func TestReplicateProviderRunPollsAndExtractsURLs(t *testing.T) {
	asset := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("asset-bytes"))
	}))
	defer asset.Close()

	var gotInput map[string]any
	var gotPath string
	gets := 0
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			gotPath = r.URL.Path
			body, _ := io.ReadAll(r.Body)
			var b map[string]any
			if err := json.Unmarshal(body, &b); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			gotInput, _ = b["input"].(map[string]any)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "run-1",
				"status": "starting",
				"urls":   map[string]string{"get": srv.URL + "/predictions/run-1"},
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
				"output": []string{asset.URL + "/a.mp4", asset.URL + "/b.mp4"},
			})
		}
	}))
	defer srv.Close()

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Run(context.Background(), &Request{
		Provider:     ProviderReplicate,
		Token:        "rtok",
		Model:        "owner/model:abc123",
		PollInterval: time.Millisecond,
	}, map[string]any{"prompt": "hi", "duration": 5})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if gotPath != "/predictions" {
		t.Errorf("versioned reference must POST /predictions, got %q", gotPath)
	}
	if gotInput["prompt"] != "hi" || gotInput["duration"] != float64(5) {
		t.Errorf("input passed through unchanged: %#v", gotInput)
	}
	if gets < 2 {
		t.Errorf("expected polling, got %d GETs", gets)
	}
	if len(res.URLs) != 2 || !strings.HasSuffix(res.URLs[1], "/b.mp4") {
		t.Errorf("urls: %#v", res.URLs)
	}
	var raw map[string]any
	if err := json.Unmarshal(res.Prediction, &raw); err != nil {
		t.Fatalf("prediction JSON: %v", err)
	}
	if raw["status"] != "succeeded" {
		t.Errorf("prediction JSON should be the upstream document: %#v", raw)
	}
	if string(res.Output) == "" {
		t.Error("output should be preserved")
	}
}

// A model whose output is not a URL (text, a blob) must still return cleanly
// with no URLs, so the caller can print the JSON.
func TestReplicateProviderRunNonURLOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "t1", "status": "succeeded",
			"output": map[string]any{"text": "hello"},
		})
	}))
	defer srv.Close()

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	res, err := p.Run(context.Background(), &Request{
		Provider: ProviderReplicate, Token: "rtok", Model: "owner/text-model",
	}, map[string]any{"prompt": "hi"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.URLs) != 0 {
		t.Errorf("no URLs expected, got %#v", res.URLs)
	}
	if !strings.Contains(string(res.Output), "hello") {
		t.Errorf("output: %s", res.Output)
	}
}

func TestReplicateProviderRunFailedPrediction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "f1", "status": "failed", "error": "boom",
		})
	}))
	defer srv.Close()

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	_, err := p.Run(context.Background(), &Request{
		Provider: ProviderReplicate, Token: "rtok", Model: "owner/model",
	}, map[string]any{"prompt": "hi"})
	if err == nil || !strings.Contains(err.Error(), "failed: boom") {
		t.Fatalf("want failed prediction error, got %v", err)
	}
}

// The model document carries enums behind allOf/$ref; -schema must resolve them.
func TestReplicateProviderSchemaResolvesRefEnums(t *testing.T) {
	doc := map[string]any{
		"latest_version": map[string]any{
			"openapi_schema": map[string]any{
				"components": map[string]any{
					"schemas": map[string]any{
						"Input": map[string]any{
							"properties": map[string]any{
								"mode": map[string]any{
									"allOf": []any{map[string]any{"$ref": "#/components/schemas/mode"}},
								},
								"temperature": map[string]any{
									"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5,
									"description": "How expressive the result is",
								},
								"active_speaker": map[string]any{"type": "boolean", "default": false},
								"audio":          map[string]any{"type": "string"},
							},
						},
						"mode": map[string]any{
							"type": "string", "enum": []any{"std", "pro"}, "default": "std",
							"description": "Motion quality",
						},
					},
				},
			},
		},
	}
	body, _ := json.Marshal(doc)

	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if r.Header.Get("Authorization") != "Bearer rtok" {
			t.Errorf("missing bearer token: %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	p := &ReplicateProvider{HTTPClient: srv.Client(), APIBase: srv.URL}
	fields, err := p.Schema(context.Background(), "rtok", "kwaivgi/kling-avatar-v2")
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if gotPath != "/models/kwaivgi/kling-avatar-v2" {
		t.Errorf("path: %q", gotPath)
	}
	if len(fields) != 4 {
		t.Fatalf("fields: %#v", fields)
	}
	// Sorted by name: active_speaker, audio, mode, temperature.
	if fields[0].Name != "active_speaker" || fields[2].Name != "mode" {
		t.Fatalf("fields must be sorted by name: %#v", fields)
	}
	mode := fields[2]
	if strings.Join(mode.Enum, "|") != "std|pro" {
		t.Errorf("enum from $ref: %#v", mode.Enum)
	}
	if mode.Type != "string" || !mode.HasDefault || mode.Default != "std" {
		t.Errorf("$ref should lend type and default: %#v", mode)
	}
	if mode.Description != "Motion quality" {
		t.Errorf("$ref should lend description: %#v", mode.Description)
	}
	if !fields[0].HasDefault || fields[0].Default != "false" {
		t.Errorf("boolean default false must render as \"false\": %#v", fields[0])
	}
	if fields[1].HasDefault {
		t.Error("a property with no default must not report one")
	}
}

func TestSchemaFieldString(t *testing.T) {
	min, max := 0.0, 1.0
	cases := []struct {
		name  string
		field SchemaField
		want  string
	}{
		{
			name:  "full",
			field: SchemaField{Name: "mode", Type: "string", Default: "std", HasDefault: true, Enum: []string{"std", "pro"}, Description: "Motion quality"},
			// Plain alphanumeric values stay unquoted; only values that would
			// read back ambiguously (whitespace, quotes, `=`) get quoted.
			want: `name=mode type=string default=std enum=[std,pro] description="Motion quality"`,
		},
		{
			name:  "plain string input",
			field: SchemaField{Name: "audio", Type: "string"},
			want:  `name=audio type=string`,
		},
		{
			name:  "numeric bounds",
			field: SchemaField{Name: "temperature", Type: "number", Minimum: &min, Maximum: &max},
			want:  `name=temperature type=number min=0 max=1`,
		},
		{
			name:  "seed keeps its letters unquoted",
			field: SchemaField{Name: "seed", Type: "integer", Default: "1", HasDefault: true},
			want:  `name=seed type=integer default=1`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.field.String(); got != tc.want {
				t.Errorf("String() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestReplicateProviderSchemaRejectsVersionPin(t *testing.T) {
	p := &ReplicateProvider{}
	_, err := p.Schema(context.Background(), "rtok", "owner/model:abc123")
	if err == nil || !strings.Contains(err.Error(), ":version") {
		t.Fatalf("want version-pin rejection, got %v", err)
	}
}

func TestParseReplicateSchemaNoInput(t *testing.T) {
	_, err := parseReplicateSchema([]byte(`{"latest_version":{"openapi_schema":{"components":{"schemas":{"Output":{"type":"string"}}}}}}`))
	if err == nil || !strings.Contains(err.Error(), "no Input schema") {
		t.Fatalf("want missing-Input error, got %v", err)
	}
}
