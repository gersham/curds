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

	input, err := buildReplicateInput(req)
	if err != nil {
		logError(req, "replicate.input_media_failed", "err", err.Error())
		return nil, err
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
	if IsVideoModel(req.Model) {
		return p.downloadVideos(ctx, req, pred.ID, urls)
	}
	return p.downloadImages(ctx, req, pred.ID, urls)
}

func buildReplicateInput(req *Request) (map[string]any, error) {
	if IsVideoModel(req.Model) {
		return buildReplicateVideoInput(req)
	}
	if IsSegmentationModel(req.Model) {
		return buildReplicateSegmentationInput(req)
	}
	if IsUpscaleModel(req.Model) {
		return buildReplicateUpscaleInput(req)
	}
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
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare input images: %w", err)
		}
		input["input_images"] = urls
	}
	return input, nil
}

// buildReplicateSegmentationInput builds the input for bria/remove-background
// (and any future SAM-style models we wire in). The model takes a single
// `image` URL/data-URL and returns one transparent PNG. No prompt, no aspect
// ratio, no quality knob.
func buildReplicateSegmentationInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare segmentation input image: %w", err)
	}
	return map[string]any{"image": urls[0]}, nil
}

// buildReplicateUpscaleInput builds the input for nightmareai/real-esrgan
// (and any future Real-ESRGAN-style super-resolution model we wire in). The
// model takes a single `image` URL/data-URL, a numeric `scale` factor, and an
// optional `face_enhance` flag, and returns one upscaled image. No prompt, no
// aspect ratio, no quality knob.
func buildReplicateUpscaleInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare upscale input image: %w", err)
	}
	scale := req.Scale
	if scale == 0 {
		scale = DefaultUpscaleScale
	}
	input := map[string]any{
		"image": urls[0],
		"scale": scale,
	}
	if req.FaceEnhance {
		input["face_enhance"] = true
	}
	return input, nil
}

func buildReplicateVideoInput(req *Request) (map[string]any, error) {
	if IsGrokImagineVideoModel(req.Model) {
		return buildReplicateGrokImagineVideoInput(req)
	}
	return buildReplicateSeedanceVideoInput(req)
}

func buildReplicateGrokImagineVideoInput(req *Request) (map[string]any, error) {
	urls, err := encodeMediaAsDataURLs(req.InputImages)
	if err != nil {
		return nil, fmt.Errorf("prepare input image: %w", err)
	}
	input := map[string]any{
		"prompt":       req.Prompt,
		"image":        urls[0],
		"duration":     req.VideoDuration,
		"resolution":   req.VideoResolution,
		"aspect_ratio": req.AspectRatio,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	return input, nil
}

func buildReplicateSeedanceVideoInput(req *Request) (map[string]any, error) {
	input := map[string]any{
		"prompt":         req.Prompt,
		"duration":       req.VideoDuration,
		"resolution":     req.VideoResolution,
		"aspect_ratio":   req.AspectRatio,
		"generate_audio": true,
	}
	if req.VideoDuration == 0 {
		input["duration"] = 5
	}
	if req.GenerateAudio != nil {
		input["generate_audio"] = *req.GenerateAudio
	}
	if req.Seed != 0 {
		input["seed"] = req.Seed
	}
	switch {
	case len(req.InputImages) == 1:
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare first frame image: %w", err)
		}
		input["image"] = urls[0]
	case len(req.InputImages) > 1:
		urls, err := encodeMediaAsDataURLs(req.InputImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		input["reference_images"] = urls
	}
	if req.LastFrameImage != "" {
		urls, err := encodeMediaAsDataURLs([]string{req.LastFrameImage})
		if err != nil {
			return nil, fmt.Errorf("prepare last frame image: %w", err)
		}
		input["last_frame_image"] = urls[0]
	}
	if len(req.ReferenceImages) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceImages)
		if err != nil {
			return nil, fmt.Errorf("prepare reference images: %w", err)
		}
		if existing, ok := input["reference_images"].([]string); ok {
			input["reference_images"] = append(existing, urls...)
		} else {
			input["reference_images"] = urls
		}
	}
	if len(req.ReferenceVideos) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceVideos)
		if err != nil {
			return nil, fmt.Errorf("prepare reference videos: %w", err)
		}
		input["reference_videos"] = urls
	}
	if len(req.ReferenceAudios) > 0 {
		urls, err := encodeMediaAsDataURLs(req.ReferenceAudios)
		if err != nil {
			return nil, fmt.Errorf("prepare reference audios: %w", err)
		}
		input["reference_audios"] = urls
	}
	return input, nil
}

func (p *ReplicateProvider) downloadImages(ctx context.Context, req *Request, id string, urls []string) (*Result, error) {
	logInfo(req, "replicate.succeeded", "id", id, "image_count", len(urls))
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

func (p *ReplicateProvider) downloadVideos(ctx context.Context, req *Request, id string, urls []string) (*Result, error) {
	logInfo(req, "replicate.succeeded", "id", id, "video_count", len(urls))
	res := &Result{Videos: make([]Video, 0, len(urls))}
	for i, u := range urls {
		b, err := p.downloadBytes(ctx, req.Token, u)
		if err != nil {
			logError(req, "video.download_failed", "index", i, "url", u, "err", err.Error())
			return nil, fmt.Errorf("download %s: %w", u, err)
		}
		logInfo(req, "video.downloaded", "index", i, "bytes", len(b), "format", req.OutputFormat)
		res.Videos = append(res.Videos, Video{Bytes: b, Format: req.OutputFormat, URL: u})
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

func encodeMediaAsDataURLs(paths []string) ([]string, error) {
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
