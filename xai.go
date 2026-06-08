package curds

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// XaiProvider talks directly to xAI's native video API
// (POST /v1/videos/generations + GET /v1/videos/{request_id}).
//
// Unlike the Replicate-hosted xai/grok-imagine-video-1.5 wrapper, the native
// API exposes the full Grok Imagine Video surface: text-to-video (image
// optional), reference images, 1080p, and durations up to 15s. It is
// asynchronous — a generation call returns a request_id that is polled until
// the video is ready, then the hosted result is downloaded.
type XaiProvider struct {
	HTTPClient *http.Client
	APIBase    string // default https://api.x.ai/v1
}

func (p *XaiProvider) Name() string { return ProviderXai }

func (p *XaiProvider) base() string {
	if p.APIBase != "" {
		return p.APIBase
	}
	return "https://api.x.ai/v1"
}

// xaiImageRef is the {url} / {file_id} object the API accepts for image inputs.
type xaiImageRef struct {
	URL string `json:"url"`
}

type xaiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *xaiError) String() string {
	if e == nil {
		return ""
	}
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

// xaiGenerateResponse is the immediate reply to a generation request.
type xaiGenerateResponse struct {
	RequestID string    `json:"request_id"`
	Error     *xaiError `json:"error"`
}

// xaiPollResponse is the body of GET /v1/videos/{request_id}.
type xaiPollResponse struct {
	Status   string `json:"status"` // pending | done | failed
	Progress int    `json:"progress"`
	Model    string `json:"model"`
	Video    *struct {
		URL      string `json:"url"`
		Duration int    `json:"duration"`
	} `json:"video"`
	Error *xaiError `json:"error"`
}

func (p *XaiProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	logInfo(req, "generation.started",
		"provider", "xai",
		"model", req.Model,
		"prompt_chars", len(req.Prompt),
		"input_images", len(req.InputImages),
		"reference_images", len(req.ReferenceImages),
		"aspect_ratio", req.AspectRatio,
		"resolution", req.VideoResolution,
	)

	body, err := buildXaiVideoBody(req)
	if err != nil {
		logError(req, "xai.input_media_failed", "err", err.Error())
		return nil, err
	}

	requestID, err := p.createGeneration(ctx, req, body)
	if err != nil {
		return nil, err
	}
	logInfo(req, "xai.created", "request_id", requestID)

	poll, err := p.waitForVideo(ctx, req, requestID)
	if err != nil {
		return nil, err
	}
	if poll.Status != "done" {
		logError(req, "xai.failed", "request_id", requestID, "status", poll.Status, "msg", poll.Error.String())
		if poll.Error != nil {
			return nil, fmt.Errorf("xai video %s: %s", poll.Status, poll.Error.String())
		}
		return nil, fmt.Errorf("xai video %s", poll.Status)
	}
	if poll.Video == nil || poll.Video.URL == "" {
		return nil, errors.New("xai video done but no video url returned")
	}

	b, err := p.downloadVideo(ctx, req, poll.Video.URL)
	if err != nil {
		logError(req, "video.download_failed", "url", poll.Video.URL, "err", err.Error())
		return nil, fmt.Errorf("download %s: %w", poll.Video.URL, err)
	}
	logInfo(req, "video.downloaded", "bytes", len(b), "format", req.OutputFormat)
	return &Result{Videos: []Video{{Bytes: b, Format: req.OutputFormat, URL: poll.Video.URL}}}, nil
}

// buildXaiVideoBody assembles the JSON body for POST /v1/videos/generations.
// The single -input-image becomes the image-to-video source; -reference-image
// entries become reference_images. An "auto" aspect ratio is omitted so the
// API derives it (from the source image, for image-to-video).
func buildXaiVideoBody(req *Request) (map[string]any, error) {
	body := map[string]any{
		"model":  req.Model,
		"prompt": req.Prompt,
	}
	duration := req.VideoDuration
	if duration == 0 {
		duration = 5
	}
	body["duration"] = duration
	if req.VideoResolution != "" {
		body["resolution"] = req.VideoResolution
	}
	if req.AspectRatio != "" && req.AspectRatio != "auto" {
		body["aspect_ratio"] = req.AspectRatio
	}
	if len(req.InputImages) == 1 {
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input image: %w", err)
		}
		body["image"] = xaiImageRef{URL: urls[0]}
	}
	if len(req.ReferenceImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		refs := make([]xaiImageRef, 0, len(urls))
		for _, u := range urls {
			refs = append(refs, xaiImageRef{URL: u})
		}
		body["reference_images"] = refs
	}
	return body, nil
}

func (p *XaiProvider) createGeneration(ctx context.Context, req *Request, body map[string]any) (string, error) {
	endpoint := p.base() + "/videos/generations"
	buf, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(buf))
	if err != nil {
		return "", err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	hreq.Header.Set("Content-Type", "application/json")

	logDebug(req, "xai.request", "endpoint", endpoint, "body", redactXaiBody(body))
	logInfo(req, "xai.request", "endpoint", endpoint)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		logError(req, "xai.transport_error", "err", err.Error())
		return "", err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	logInfo(req, "xai.response", "status_code", resp.StatusCode, "bytes", len(rb))
	if resp.StatusCode >= 400 {
		logError(req, "xai.api_error", "status_code", resp.StatusCode, "body", truncate(string(rb), 500))
		return "", fmt.Errorf("xai API %d: %s", resp.StatusCode, string(rb))
	}
	var out xaiGenerateResponse
	if err := json.Unmarshal(rb, &out); err != nil {
		return "", fmt.Errorf("decode generation response: %w (body=%s)", err, string(rb))
	}
	if out.Error != nil {
		return "", fmt.Errorf("xai: %s", out.Error.String())
	}
	if out.RequestID == "" {
		return "", fmt.Errorf("xai returned no request_id: %s", string(rb))
	}
	return out.RequestID, nil
}

func (p *XaiProvider) waitForVideo(ctx context.Context, req *Request, requestID string) (*xaiPollResponse, error) {
	getURL := fmt.Sprintf("%s/videos/%s", p.base(), url.PathEscape(requestID))
	for {
		poll, err := p.fetchVideo(ctx, req, getURL)
		if err != nil {
			logError(req, "xai.poll_failed", "request_id", requestID, "err", err.Error())
			return nil, err
		}
		if isXaiTerminalStatus(poll.Status) {
			logInfo(req, "xai.terminal", "request_id", requestID, "status", poll.Status)
			return poll, nil
		}
		logInfo(req, "xai.polling", "request_id", requestID, "status", poll.Status, "progress", poll.Progress, "interval", req.PollInterval.String())
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(req.PollInterval):
		}
	}
}

func (p *XaiProvider) fetchVideo(ctx context.Context, req *Request, getURL string) (*xaiPollResponse, error) {
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
		return nil, fmt.Errorf("get video %d: %s", resp.StatusCode, string(rb))
	}
	var poll xaiPollResponse
	if err := json.Unmarshal(rb, &poll); err != nil {
		return nil, fmt.Errorf("decode poll response: %w (body=%s)", err, string(rb))
	}
	return &poll, nil
}

func (p *XaiProvider) downloadVideo(ctx context.Context, req *Request, rawURL string) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	// Only attach the bearer token when the asset is on an xAI-controlled host,
	// so a redirect to a third party can't leak the credential.
	if req.Token != "" && isXaiHost(rawURL) {
		hreq.Header.Set("Authorization", "Bearer "+req.Token)
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

// isXaiHost reports whether the URL points at an xAI-controlled host. Matches
// on the parsed hostname (not a substring) so a crafted URL can't leak the
// bearer token. Covers x.ai and its asset host vidgen.x.ai.
func isXaiHost(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil || u == nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}
	if host == "x.ai" {
		return true
	}
	return strings.HasSuffix(host, ".x.ai")
}

func isXaiTerminalStatus(s string) bool {
	switch s {
	case "done", "failed":
		return true
	}
	return false
}

// redactXaiBody renders the request body for logging. Image inputs can be
// large base64 data URIs, so they're elided rather than dumped.
func redactXaiBody(body map[string]any) string {
	clone := make(map[string]any, len(body))
	for k, v := range body {
		switch k {
		case "image":
			clone[k] = "[image]"
		case "reference_images":
			clone[k] = fmt.Sprintf("[%d reference image(s)]", lenOf(v))
		default:
			clone[k] = v
		}
	}
	b, err := json.Marshal(clone)
	if err != nil {
		return "<unloggable body>"
	}
	return string(b)
}

func lenOf(v any) int {
	if s, ok := v.([]xaiImageRef); ok {
		return len(s)
	}
	return 0
}
