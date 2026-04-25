package curds

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"path/filepath"
	"strings"
)

// OpenAIProvider talks directly to OpenAI's Image API.
type OpenAIProvider struct {
	HTTPClient *http.Client
	APIBase    string // default https://api.openai.com/v1
}

func (p *OpenAIProvider) Name() string { return ProviderOpenAI }

func (p *OpenAIProvider) base() string {
	if p.APIBase != "" {
		return p.APIBase
	}
	return "https://api.openai.com/v1"
}

type openAIImageResponse struct {
	Data []struct {
		B64JSON       string `json:"b64_json"`
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (p *OpenAIProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	size := ResolveSize(req)
	logInfo(req, "generation.started",
		"provider", "openai",
		"model", req.Model,
		"size", size,
		"prompt_chars", len(req.Prompt),
		"num_images", req.NumImages,
		"input_images", len(req.InputImages),
	)
	if len(req.InputImages) > 0 || req.Mask != "" {
		return p.callEdits(ctx, req, size)
	}
	return p.callGenerations(ctx, req, size)
}

func (p *OpenAIProvider) callGenerations(ctx context.Context, req *Request, size string) (*Result, error) {
	body := map[string]any{
		"model":         req.Model,
		"prompt":        req.Prompt,
		"n":             req.NumImages,
		"size":          size,
		"quality":       req.Quality,
		"background":    req.Background,
		"moderation":    req.Moderation,
		"output_format": req.OutputFormat,
	}
	if req.OutputFormat == "jpeg" || req.OutputFormat == "webp" {
		body["output_compression"] = req.OutputCompression
	}
	if req.User != "" {
		body["user"] = req.User
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	endpoint := p.base() + "/images/generations"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	hreq.Header.Set("Content-Type", "application/json")
	logInfo(req, "openai.request", "endpoint", endpoint, "kind", "generations")
	logDebug(req, "openai.request_body", "body", truncate(string(buf), 500))
	return p.do(hreq, req)
}

func (p *OpenAIProvider) callEdits(ctx context.Context, req *Request, size string) (*Result, error) {
	if len(req.InputImages) == 0 {
		return nil, errors.New("openai edits endpoint requires at least one input image")
	}

	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)

	go func() {
		defer pw.Close()
		defer mw.Close()
		fail := func(err error) { pw.CloseWithError(err) }

		fields := map[string]string{
			"model":         req.Model,
			"prompt":        req.Prompt,
			"n":             fmt.Sprintf("%d", req.NumImages),
			"size":          size,
			"quality":       req.Quality,
			"background":    req.Background,
			"moderation":    req.Moderation,
			"output_format": req.OutputFormat,
		}
		for k, v := range fields {
			if err := mw.WriteField(k, v); err != nil {
				fail(err)
				return
			}
		}
		if req.OutputFormat == "jpeg" || req.OutputFormat == "webp" {
			if err := mw.WriteField("output_compression", fmt.Sprintf("%d", req.OutputCompression)); err != nil {
				fail(err)
				return
			}
		}
		if req.User != "" {
			if err := mw.WriteField("user", req.User); err != nil {
				fail(err)
				return
			}
		}
		for _, ip := range req.InputImages {
			if err := writeImagePart(mw, "image[]", ip); err != nil {
				fail(fmt.Errorf("image %s: %w", ip, err))
				return
			}
		}
		if req.Mask != "" {
			if err := writeImagePart(mw, "mask", req.Mask); err != nil {
				fail(fmt.Errorf("mask %s: %w", req.Mask, err))
				return
			}
		}
	}()

	endpoint := p.base() + "/images/edits"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, pr)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Authorization", "Bearer "+req.Token)
	hreq.Header.Set("Content-Type", mw.FormDataContentType())
	logInfo(req, "openai.request", "endpoint", endpoint, "kind", "edits", "input_images", len(req.InputImages))
	return p.do(hreq, req)
}

func writeImagePart(mw *multipart.Writer, field, path string) error {
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return fmt.Errorf("openai edits requires local file paths, got URL: %s", path)
	}
	if strings.HasPrefix(path, "data:") {
		return fmt.Errorf("openai edits requires local file paths, got data URL")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	mt := detectMime(path, data)
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q; filename=%q`, field, filepath.Base(path)))
	h.Set("Content-Type", mt)
	pw, err := mw.CreatePart(h)
	if err != nil {
		return err
	}
	_, err = pw.Write(data)
	return err
}

func (p *OpenAIProvider) do(hreq *http.Request, req *Request) (*Result, error) {
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		logError(req, "openai.transport_error", "err", err.Error())
		return nil, err
	}
	defer resp.Body.Close()
	rb, _ := io.ReadAll(resp.Body)
	logInfo(req, "openai.response", "status_code", resp.StatusCode, "bytes", len(rb))
	if resp.StatusCode >= 400 {
		logError(req, "openai.api_error", "status_code", resp.StatusCode, "body", truncate(string(rb), 500))
		return nil, fmt.Errorf("openai API %d: %s", resp.StatusCode, string(rb))
	}
	var out openAIImageResponse
	if err := json.Unmarshal(rb, &out); err != nil {
		logError(req, "openai.decode_error", "err", err.Error())
		return nil, fmt.Errorf("decode openai response: %w (body=%s)", err, string(rb))
	}
	if out.Error != nil {
		logError(req, "openai.api_error", "msg", out.Error.Message, "type", out.Error.Type, "code", out.Error.Code)
		return nil, fmt.Errorf("openai: %s", out.Error.Message)
	}

	res := &Result{Images: make([]Image, 0, len(out.Data))}
	for i, d := range out.Data {
		img := Image{Format: req.OutputFormat, RevisedPrompt: d.RevisedPrompt}
		switch {
		case d.B64JSON != "":
			b, err := base64.StdEncoding.DecodeString(d.B64JSON)
			if err != nil {
				logError(req, "openai.b64_decode_error", "index", i, "err", err.Error())
				return nil, fmt.Errorf("decode b64 image %d: %w", i, err)
			}
			img.Bytes = b
			logInfo(req, "image.received", "index", i, "bytes", len(b), "format", req.OutputFormat, "has_revised_prompt", d.RevisedPrompt != "")
		case d.URL != "":
			b, err := downloadOpenAIImage(hreq.Context(), p.HTTPClient, d.URL)
			if err != nil {
				logError(req, "image.download_failed", "index", i, "url", d.URL, "err", err.Error())
				return nil, fmt.Errorf("download image %d: %w", i, err)
			}
			img.Bytes = b
			logInfo(req, "image.downloaded", "index", i, "bytes", len(b), "format", req.OutputFormat)
		default:
			return nil, fmt.Errorf("image %d had neither b64_json nor url", i)
		}
		res.Images = append(res.Images, img)
	}
	if len(res.Images) == 0 {
		return nil, fmt.Errorf("openai returned no images: %s", string(rb))
	}
	return res, nil
}

func downloadOpenAIImage(ctx context.Context, hc *http.Client, url string) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := httpClientOrDefault(hc).Do(hreq)
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
