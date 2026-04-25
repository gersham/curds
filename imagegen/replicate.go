package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ReplicateProvider talks to the Replicate REST API.
//
// For an official model like "owner/name" it uses the dedicated
// /v1/models/{owner}/{name}/predictions endpoint. For a versioned
// "owner/name:version" reference it uses /v1/predictions.
type ReplicateProvider struct {
	HTTPClient *http.Client
	APIBase    string // default https://api.replicate.com/v1
}

func (p *ReplicateProvider) Name() string { return ProviderReplicate }

func (p *ReplicateProvider) base() string {
	if p.APIBase != "" {
		return p.APIBase
	}
	return "https://api.replicate.com/v1"
}

type replicatePrediction struct {
	ID     string            `json:"id"`
	Status string            `json:"status"`
	Output json.RawMessage   `json:"output"`
	Error  any               `json:"error"`
	Logs   string            `json:"logs"`
	URLs   map[string]string `json:"urls"`
}

func (p *ReplicateProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	input := map[string]any{
		"prompt":           req.Prompt,
		"aspect_ratio":     req.AspectRatio,
		"quality":          req.Quality,
		"number_of_images": req.NumImages,
		"output_format":    req.OutputFormat,
		"background":       req.Background,
		"moderation":       req.Moderation,
	}
	if req.ReplicateBYOKey != "" {
		input["openai_api_key"] = req.ReplicateBYOKey
	}
	if len(req.InputImages) > 0 {
		urls, err := encodeInputImagesAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["input_images"] = urls
	}

	pred, err := p.createPrediction(ctx, req, input)
	if err != nil {
		return nil, err
	}
	pred, err = p.waitForPrediction(ctx, req, pred)
	if err != nil {
		return nil, err
	}
	if pred.Status != "succeeded" {
		return nil, fmt.Errorf("prediction %s: %s", pred.Status, formatErr(pred.Error))
	}

	urls, err := extractOutputURLs(pred.Output)
	if err != nil {
		return nil, fmt.Errorf("parse output: %w", err)
	}
	if len(urls) == 0 {
		return nil, errors.New("prediction succeeded but produced no output URLs")
	}

	res := &Result{Images: make([]Image, 0, len(urls))}
	for _, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		res.Images = append(res.Images, Image{Bytes: b, Format: req.OutputFormat})
	}
	return res, nil
}

func (p *ReplicateProvider) createPrediction(ctx context.Context, req *Request, input map[string]any) (*replicatePrediction, error) {
	body := map[string]any{"input": input}
	var endpoint string
	if idx := strings.Index(req.Model, ":"); idx >= 0 {
		body["version"] = req.Model[idx+1:]
		endpoint = p.base() + "/predictions"
	} else {
		endpoint = fmt.Sprintf("%s/models/%s/predictions", p.base(), req.Model)
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Prefer", "wait=60")

	logf(req.Logger, "→ POST %s\n%s\n", endpoint, string(buf))
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	logf(req.Logger, "← %d (%d bytes)\n", resp.StatusCode, len(rb))
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("replicate API %d: %s", resp.StatusCode, string(rb))
	}
	var pred replicatePrediction
	if err := json.Unmarshal(rb, &pred); err != nil {
		return nil, fmt.Errorf("decode prediction: %w (body=%s)", err, string(rb))
	}
	return &pred, nil
}

func (p *ReplicateProvider) waitForPrediction(ctx context.Context, req *Request, pred *replicatePrediction) (*replicatePrediction, error) {
	for !isTerminalStatus(pred.Status) {
		logf(req.Logger, "status: %s (sleeping %s)\n", pred.Status, req.PollInterval)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(req.PollInterval):
		}
		next, err := p.fetchPrediction(ctx, req, pred.URLs["get"])
		if err != nil {
			return nil, err
		}
		pred = next
	}
	logf(req.Logger, "status: %s\n", pred.Status)
	return pred, nil
}

func (p *ReplicateProvider) fetchPrediction(ctx context.Context, req *Request, getURL string) (*replicatePrediction, error) {
	if getURL == "" {
		return nil, errors.New("missing prediction get URL")
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, getURL, nil)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("get prediction %d: %s", resp.StatusCode, string(rb))
	}
	var pred replicatePrediction
	if err := json.Unmarshal(rb, &pred); err != nil {
		return nil, err
	}
	return &pred, nil
}

func (p *ReplicateProvider) downloadBytes(ctx context.Context, token, url string) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" && (strings.Contains(url, "replicate.delivery") || strings.Contains(url, "api.replicate.com")) {
		hreq.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(b))
	}
	return io.ReadAll(resp.Body)
}

func isTerminalStatus(s string) bool {
	switch s {
	case "succeeded", "failed", "canceled":
		return true
	}
	return false
}

func extractOutputURLs(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []string{s}, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var anyArr []any
	if err := json.Unmarshal(raw, &anyArr); err == nil {
		out := make([]string, 0, len(anyArr))
		for _, x := range anyArr {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported output shape: %s", string(raw))
}

func encodeInputImagesAsDataURLs(paths []string) ([]string, error) {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if strings.HasPrefix(p, "http://") || strings.HasPrefix(p, "https://") || strings.HasPrefix(p, "data:") {
			out = append(out, p)
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		mt := detectMime(p, data)
		out = append(out, fmt.Sprintf("data:%s;base64,%s", mt, base64.StdEncoding.EncodeToString(data)))
	}
	return out, nil
}

func formatErr(e any) string {
	if e == nil {
		return ""
	}
	if s, ok := e.(string); ok {
		return s
	}
	b, _ := json.Marshal(e)
	return string(b)
}
