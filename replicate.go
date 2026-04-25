package curds

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
	logInfo(req, "generation.started",
		"provider", "replicate",
		"model", req.Model,
		"prompt_chars", len(req.Prompt),
		"num_images", req.NumImages,
		"aspect_ratio", req.AspectRatio,
	)

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
			logError(req, "replicate.input_image_failed", "err", err.Error())
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["input_images"] = urls
	}

	pred, err := p.createPrediction(ctx, req, input)
	if err != nil {
		return nil, err
	}
	logInfo(req, "replicate.created", "id", pred.ID, "status", pred.Status)

	pred, err = p.waitForPrediction(ctx, req, pred)
	if err != nil {
		return nil, err
	}
	if pred.Status != "succeeded" {
		logError(req, "replicate.failed", "id", pred.ID, "status", pred.Status, "msg", formatErr(pred.Error))
		return nil, fmt.Errorf("prediction %s: %s", pred.Status, formatErr(pred.Error))
	}

	urls, err := extractOutputURLs(pred.Output)
	if err != nil {
		return nil, fmt.Errorf("parse output: %w", err)
	}
	if len(urls) == 0 {
		return nil, errors.New("prediction succeeded but produced no output URLs")
	}
	logInfo(req, "replicate.succeeded", "id", pred.ID, "image_count", len(urls))

	res := &Result{Images: make([]Image, 0, len(urls))}
	for i, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			logError(req, "image.download_failed", "index", i, "url", u, "err", err.Error())
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		logInfo(req, "image.downloaded", "index", i, "bytes", len(b), "format", req.OutputFormat)
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

	logDebug(req, "replicate.request", "endpoint", endpoint, "body", redactRequestBody(body))
	logInfo(req, "replicate.request", "endpoint", endpoint)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		logError(req, "replicate.transport_error", "err", err.Error())
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	logInfo(req, "replicate.response", "status_code", resp.StatusCode, "bytes", len(rb))
	if resp.StatusCode >= 400 {
		logError(req, "replicate.api_error", "status_code", resp.StatusCode, "body", truncate(string(rb), 500))
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
		logInfo(req, "replicate.polling", "id", pred.ID, "status", pred.Status, "interval", req.PollInterval.String())
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(req.PollInterval):
		}
		next, err := p.fetchPrediction(ctx, req, pred.URLs["get"])
		if err != nil {
			logError(req, "replicate.poll_failed", "id", pred.ID, "err", err.Error())
			return nil, err
		}
		pred = next
	}
	logInfo(req, "replicate.terminal", "id", pred.ID, "status", pred.Status)
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

func (p *ReplicateProvider) downloadBytes(ctx context.Context, token, rawURL string) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	if token != "" && isReplicateHost(rawURL) {
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

// isReplicateHost reports whether the URL points at a Replicate-controlled
// host. It compares the parsed hostname instead of substring-matching the
// raw URL — substring matches let an attacker craft URLs like
// https://evil.example/?replicate.delivery that would otherwise leak the
// bearer token.
func isReplicateHost(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil || u == nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}
	switch host {
	case "api.replicate.com", "replicate.com", "replicate.delivery":
		return true
	}
	return strings.HasSuffix(host, ".replicate.com") || strings.HasSuffix(host, ".replicate.delivery")
}

// redactRequestBody returns a JSON-marshaled view of the request body with
// secret-bearing fields scrubbed for logging.
func redactRequestBody(body map[string]any) string {
	clone := make(map[string]any, len(body))
	for k, v := range body {
		clone[k] = v
	}
	if input, ok := clone["input"].(map[string]any); ok {
		safe := make(map[string]any, len(input))
		for k, v := range input {
			if k == "openai_api_key" {
				safe[k] = "[REDACTED]"
				continue
			}
			safe[k] = v
		}
		clone["input"] = safe
	}
	b, err := json.Marshal(clone)
	if err != nil {
		return "<unloggable body>"
	}
	return string(b)
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
