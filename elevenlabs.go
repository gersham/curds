package curds

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

// DefaultElevenLabsAPIBase is the ElevenLabs API root. Its paths carry their
// own version (/v1/text-to-speech, /v1/music, /v2/voices), so unlike the other
// providers the base has no /v1 suffix.
const DefaultElevenLabsAPIBase = "https://api.elevenlabs.io"

// POST /v1/music takes music_length_ms between 3000 and 600000.
const (
	MinElevenLabsMusicSeconds = 3
	MaxElevenLabsMusicSeconds = 600
)

// errDirectOnlySettings is the rejection for -similarity / -speaker-boost on a
// model that is not on the direct ElevenLabs route.
const errDirectOnlySettings = "-similarity and -speaker-boost need the direct ElevenLabs route: -model tts-elevenlabs with an elevenlabs token (ELEVENLABS_API_KEY or [tokens] elevenlabs)"

// ElevenLabsProvider talks directly to the ElevenLabs API: text to speech
// (POST /v1/text-to-speech/{voice_id}), music (POST /v1/music), and voice
// discovery (GET /v2/voices, GET /v1/shared-voices). Every request carries the
// key in the xi-api-key header.
type ElevenLabsProvider struct {
	HTTPClient *http.Client
	APIBase    string // default https://api.elevenlabs.io
}

func (p *ElevenLabsProvider) Name() string { return ProviderElevenLabs }

func (p *ElevenLabsProvider) base() string {
	if p.APIBase != "" {
		return strings.TrimSuffix(p.APIBase, "/")
	}
	return DefaultElevenLabsAPIBase
}

// Generate renders req.Prompt as speech or music, picked by the model id.
func (p *ElevenLabsProvider) Generate(ctx context.Context, req *Request) (*Result, error) {
	switch {
	case IsElevenLabsDirectMusicModel(req.Model):
		return p.compose(ctx, req)
	case IsElevenLabsDirectTTSModel(req.Model):
		return p.speak(ctx, req)
	}
	return nil, fmt.Errorf("unsupported elevenlabs model %q", req.Model)
}

// elevenLabsTTSSpec is what one ElevenLabs TTS model accepts: its per-request
// character cap and which voice_settings beyond stability it honors. Sources:
// GET /v1/models (maximum_text_length_per_request, can_use_style,
// can_use_speaker_boost), the Eleven v4 page ("Both variants use two voice
// settings: Stability and Similarity"), the TTS product guide (no speed,
// similarity or speaker boost on Eleven v3), and the Flash v2.5 voice settings
// (stability, similarity_boost, speed).
type elevenLabsTTSSpec struct {
	maxChars     int
	similarity   bool
	style        bool
	speed        bool
	speakerBoost bool
}

var elevenLabsTTSModels = map[string]elevenLabsTTSSpec{
	"eleven_v4":              {maxChars: 10000, similarity: true},
	"eleven_v4_turbo":        {maxChars: 10000, similarity: true},
	"eleven_v3":              {maxChars: 5000},
	"eleven_multilingual_v2": {maxChars: 10000, similarity: true, style: true, speed: true, speakerBoost: true},
	"eleven_flash_v2_5":      {maxChars: 40000, similarity: true, speed: true},
	"eleven_flash_v2":        {maxChars: 30000, similarity: true, speed: true},
}

// ElevenLabsMaxChars returns the per-request character cap of an ElevenLabs
// TTS model, or 0 when curds does not know the model (the API enforces it).
func ElevenLabsMaxChars(model string) int {
	return elevenLabsTTSModels[strings.ToLower(strings.TrimSpace(model))].maxChars
}

// CheckElevenLabsVoiceSettings validates the direct route's voice settings:
// the 0-1 ranges, speed 0.7-1.2, and — for a model curds knows — whether the
// model takes the knob at all, so -style on eleven_v4 fails locally instead of
// being ignored. Stability applies to every model. A model id curds does not
// know passes everything through for the API to judge.
func CheckElevenLabsVoiceSettings(model string, stability, similarity, style *float64, speed float64, speakerBoost bool) error {
	for _, k := range []struct {
		flag string
		v    *float64
	}{{"-stability", stability}, {"-similarity", similarity}, {"-style", style}} {
		if k.v != nil && (*k.v < 0 || *k.v > 1) {
			return fmt.Errorf("%s must be 0-1 for the direct ElevenLabs route, got %g", k.flag, *k.v)
		}
	}
	if speed != 0 && (speed < 0.7 || speed > 1.2) {
		return fmt.Errorf("-speed must be 0.7-1.2 for the direct ElevenLabs route, got %g", speed)
	}
	spec, known := elevenLabsTTSModels[strings.ToLower(strings.TrimSpace(model))]
	if !known {
		return nil
	}
	var unsupported []string
	if similarity != nil && !spec.similarity {
		unsupported = append(unsupported, "-similarity")
	}
	if style != nil && !spec.style {
		unsupported = append(unsupported, "-style")
	}
	if speed != 0 && !spec.speed {
		unsupported = append(unsupported, "-speed")
	}
	if speakerBoost && !spec.speakerBoost {
		unsupported = append(unsupported, "-speaker-boost")
	}
	if len(unsupported) == 0 {
		return nil
	}
	takes := []string{"-stability"}
	if spec.similarity {
		takes = append(takes, "-similarity")
	}
	if spec.style {
		takes = append(takes, "-style")
	}
	if spec.speed {
		takes = append(takes, "-speed")
	}
	if spec.speakerBoost {
		takes = append(takes, "-speaker-boost")
	}
	verb := "is"
	if len(unsupported) > 1 {
		verb = "are"
	}
	return fmt.Errorf("%s %s not supported by %s, which takes %s; add -tts-model eleven_multilingual_v2 for -similarity, -style, -speed and -speaker-boost",
		joinAnd(unsupported), verb, model, joinAnd(takes))
}

// joinAnd renders a list as prose: "a", "a and b", "a, b and c".
func joinAnd(items []string) string {
	if len(items) <= 1 {
		return strings.Join(items, "")
	}
	return strings.Join(items[:len(items)-1], ", ") + " and " + items[len(items)-1]
}

// elevenLabsVoiceSettings builds the voice_settings object from the knobs the
// caller set. Unset knobs are omitted so the voice's stored settings apply;
// nil means no voice_settings at all.
func elevenLabsVoiceSettings(req *Request) map[string]any {
	settings := map[string]any{}
	if req.Stability != nil {
		settings["stability"] = *req.Stability
	}
	if req.SimilarityBoost != nil {
		settings["similarity_boost"] = *req.SimilarityBoost
	}
	if req.Style != nil {
		settings["style"] = *req.Style
	}
	if req.Speed > 0 {
		settings["speed"] = req.Speed
	}
	if req.SpeakerBoost != nil {
		settings["use_speaker_boost"] = *req.SpeakerBoost
	}
	if len(settings) == 0 {
		return nil
	}
	return settings
}

// elevenLabsSpeechFormats is the output_format ladder for text to speech, best
// first. mp3_44100_192 needs the Creator tier and wav_44100 the Pro tier; a
// plan below that answers 403 output_format_not_allowed and curds steps down to
// the next entry.
func elevenLabsSpeechFormats(container string) []string {
	if container == "wav" {
		return []string{"wav_44100", "wav_24000"}
	}
	return []string{"mp3_44100_192", "mp3_44100_128"}
}

// elevenLabsMusicFormats is the output_format ladder for POST /v1/music, best
// first. The music endpoint has no wav format, so a wav -output gets the mp3
// and the CLI transcodes it locally.
func elevenLabsMusicFormats() []string {
	return []string{"mp3_48000_320", "mp3_48000_192", "mp3_44100_128"}
}

// elevenLabsContainer maps an output_format (codec_samplerate_bitrate) onto
// curds' container name.
func elevenLabsContainer(format string) string {
	codec, _, _ := strings.Cut(format, "_")
	return codec
}

// speak renders req.Prompt through POST /v1/text-to-speech/{voice_id}. The
// response body is the audio file itself.
func (p *ElevenLabsProvider) speak(ctx context.Context, req *Request) (*Result, error) {
	logInfo(req, "generation.started",
		"provider", "elevenlabs",
		"model", req.Model,
		"text_chars", len(req.Prompt),
		"voice", req.Voice,
		"format", req.OutputFormat,
	)
	voiceID, err := p.resolveVoice(ctx, req)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"text":     req.Prompt,
		"model_id": req.Model,
	}
	if settings := elevenLabsVoiceSettings(req); settings != nil {
		body["voice_settings"] = settings
	}
	audio, format, _, err := p.postAudio(ctx, req, "speech", "/v1/text-to-speech/"+url.PathEscape(voiceID), body, elevenLabsSpeechFormats(req.OutputFormat))
	if err != nil {
		return nil, err
	}
	container := elevenLabsContainer(format)
	logInfo(req, "audio.received", "bytes", len(audio), "format", container, "output_format", format)
	return &Result{Audios: []Audio{{Bytes: audio, Format: container}}}, nil
}

// compose renders req.Prompt through POST /v1/music: the prompt, the music
// model id, force_instrumental (curds' default for -model music is
// instrumental), and an exact music_length_ms when -duration is set (the model
// picks a length from the prompt otherwise).
func (p *ElevenLabsProvider) compose(ctx context.Context, req *Request) (*Result, error) {
	logInfo(req, "generation.started",
		"provider", "elevenlabs",
		"model", req.Model,
		"prompt_chars", len(req.Prompt),
		"duration_s", req.Duration,
		"format", req.OutputFormat,
	)
	body := map[string]any{
		"prompt":             req.Prompt,
		"model_id":           req.Model,
		"force_instrumental": instrumentalValue(req, true),
	}
	if req.Duration > 0 {
		body["music_length_ms"] = int(math.Round(req.Duration * 1000))
	}
	audio, format, header, err := p.postAudio(ctx, req, "music", "/v1/music", body, elevenLabsMusicFormats())
	if err != nil {
		return nil, err
	}
	container := elevenLabsContainer(format)
	logInfo(req, "audio.received", "bytes", len(audio), "format", container, "output_format", format, "song_id", header.Get("song-id"))
	return &Result{Audios: []Audio{{Bytes: audio, Format: container}}}, nil
}

// postAudio POSTs body to path once per output_format in formats, best first,
// stepping down only when the API says the account's plan does not allow that
// format. It returns the audio bytes, the accepted format, and the response
// headers.
func (p *ElevenLabsProvider) postAudio(ctx context.Context, req *Request, kind, path string, body map[string]any, formats []string) ([]byte, string, http.Header, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, "", nil, err
	}
	endpoint := p.base() + path
	for i, format := range formats {
		hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"?output_format="+url.QueryEscape(format), bytes.NewReader(buf))
		if err != nil {
			return nil, "", nil, err
		}
		hreq.Header.Set("xi-api-key", req.Token)
		hreq.Header.Set("Content-Type", "application/json")
		logInfo(req, "elevenlabs.request", "endpoint", endpoint, "kind", kind, "output_format", format)
		logDebug(req, "elevenlabs.request_body", "body", truncate(string(buf), 500))

		resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
		if err != nil {
			logError(req, "elevenlabs.transport_error", "err", err.Error())
			return nil, "", nil, err
		}
		rb, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		logInfo(req, "elevenlabs.response", "status_code", resp.StatusCode, "bytes", len(rb))
		if resp.StatusCode < 400 {
			if len(rb) == 0 {
				return nil, "", nil, errors.New("elevenlabs returned no audio")
			}
			return rb, format, resp.Header, nil
		}
		apiErr := parseElevenLabsError(rb)
		if i+1 < len(formats) && apiErr.formatNotAllowed() {
			logInfo(req, "elevenlabs.format_fallback", "from", format, "to", formats[i+1], "reason", apiErr.Message)
			continue
		}
		logError(req, "elevenlabs.api_error", "status_code", resp.StatusCode, "body", truncate(string(rb), 500))
		return nil, "", nil, fmt.Errorf("elevenlabs API %d: %s", resp.StatusCode, apiErr.text(rb))
	}
	return nil, "", nil, errors.New("elevenlabs: no output format to request")
}

// elevenLabsErrorDetail is the `detail` object of an ElevenLabs error body.
// Validation failures (422) send a list of {loc, msg} instead, which is folded
// into Message.
type elevenLabsErrorDetail struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Param   string `json:"param"`
}

func parseElevenLabsError(body []byte) elevenLabsErrorDetail {
	var env struct {
		Detail json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(body, &env) != nil || len(env.Detail) == 0 {
		return elevenLabsErrorDetail{}
	}
	var d elevenLabsErrorDetail
	if json.Unmarshal(env.Detail, &d) == nil {
		return d
	}
	var list []struct {
		Msg string `json:"msg"`
	}
	if json.Unmarshal(env.Detail, &list) == nil {
		msgs := make([]string, 0, len(list))
		for _, item := range list {
			if item.Msg != "" {
				msgs = append(msgs, item.Msg)
			}
		}
		d.Message = strings.Join(msgs, "; ")
		return d
	}
	var s string
	if json.Unmarshal(env.Detail, &s) == nil {
		d.Message = s
	}
	return d
}

// formatNotAllowed reports whether the error is the API refusing an
// output_format for the account's plan, e.g. 403 {"code":
// "subscription_required", "status": "output_format_not_allowed", "message":
// "Output format 'mp3_44100_192' is only available on the Creator tier and
// above."}.
func (d elevenLabsErrorDetail) formatNotAllowed() bool {
	if d.Status == "output_format_not_allowed" || d.Param == "output_format" || d.Code == "invalid_output_format" {
		return true
	}
	return (d.Code == "subscription_required" || d.Code == "feature_not_available") &&
		strings.Contains(strings.ToLower(d.Message), "output format")
}

// text is the human-readable error: the API's message when it sent one, the
// raw body otherwise.
func (d elevenLabsErrorDetail) text(body []byte) string {
	if d.Message != "" {
		return d.Message
	}
	return strings.TrimSpace(string(body))
}

// elevenLabsVoiceIDPattern matches an ElevenLabs voice id: 20 letters and
// digits (JBFqnCBsd6RMkjVDRZzb).
var elevenLabsVoiceIDPattern = regexp.MustCompile(`^[A-Za-z0-9]{20}$`)

// resolveVoice turns -voice into a voice id. An id is used as-is (any account,
// premade, or library voice); anything else is a name looked up among the
// account's voices with GET /v2/voices.
func (p *ElevenLabsProvider) resolveVoice(ctx context.Context, req *Request) (string, error) {
	voice := strings.TrimSpace(req.Voice)
	if elevenLabsVoiceIDPattern.MatchString(voice) {
		return voice, nil
	}
	voices, err := p.AccountVoices(ctx, req.Token, VoiceQuery{Search: voice, Limit: 100})
	if err != nil {
		logError(req, "elevenlabs.voice_lookup_failed", "voice", voice, "err", err.Error())
		return "", fmt.Errorf("resolve voice %q: %w", voice, err)
	}
	id, err := matchVoiceName(voice, voices)
	if err != nil {
		return "", err
	}
	logInfo(req, "elevenlabs.voice_resolved", "voice", voice, "id", id)
	return id, nil
}

// matchVoiceName picks the voice called name, case-insensitively. Premade
// voices carry a tagline ("George - Warm, Captivating Storyteller"), so the part
// before " - " counts too, but an exact full-name match wins over it. Several
// voices at the winning rank are an error that lists them.
func matchVoiceName(name string, voices []Voice) (string, error) {
	want := strings.ToLower(strings.TrimSpace(name))
	var exact, short []Voice
	for _, v := range voices {
		n := strings.ToLower(strings.TrimSpace(v.Name))
		if n == want {
			exact = append(exact, v)
			continue
		}
		if head, _, ok := strings.Cut(n, " - "); ok && strings.TrimSpace(head) == want {
			short = append(short, v)
		}
	}
	for _, set := range [][]Voice{exact, short} {
		switch len(set) {
		case 0:
			continue
		case 1:
			return set[0].ID, nil
		}
		names := make([]string, 0, len(set))
		for _, v := range set {
			names = append(names, fmt.Sprintf("%s (%s)", strings.TrimSpace(v.Name), v.ID))
		}
		return "", fmt.Errorf("voice %q matches %d voices in your ElevenLabs account: %s; pass the id to -voice", name, len(set), strings.Join(names, ", "))
	}
	return "", fmt.Errorf("no voice named %q in your ElevenLabs account; find one with curds voices -search %q and pass its id to -voice, or add -provider replicate for Replicate's elevenlabs/v3 voices", name, name)
}

// Voice is one ElevenLabs voice: from the account (GET /v2/voices) or from the
// shared voice library (GET /v1/shared-voices).
type Voice struct {
	Source      string // "account" or "library"
	ID          string
	Name        string
	Accent      string
	Age         string
	Gender      string
	Description string
	Category    string
}

// String renders the voice as one logfmt line, the shape `curds voices`
// prints.
func (v Voice) String() string {
	return formatSchemaKV([]any{
		"source", v.Source,
		"id", v.ID,
		"name", strings.TrimSpace(v.Name),
		"accent", v.Accent,
		"age", v.Age,
		"gender", v.Gender,
		"description", strings.Join(strings.Fields(v.Description), " "),
	})
}

// VoiceQuery filters a voice listing. Both endpoints filter on the same labels
// server-side. Limit is the page size (1-100; 0 = 30).
type VoiceQuery struct {
	Search string
	Accent string
	Gender string
	Age    string
	Limit  int
}

func (q VoiceQuery) values() url.Values {
	v := url.Values{}
	for key, val := range map[string]string{"search": q.Search, "accent": q.Accent, "gender": q.Gender, "age": q.Age} {
		if s := strings.TrimSpace(val); s != "" {
			v.Set(key, s)
		}
	}
	limit := q.Limit
	if limit <= 0 {
		limit = 30
	}
	if limit > 100 {
		limit = 100
	}
	v.Set("page_size", strconv.Itoa(limit))
	return v
}

// AccountVoices lists the voices the account can use (its own, saved, and
// premade) via GET /v2/voices, filtered server-side by q.
func (p *ElevenLabsProvider) AccountVoices(ctx context.Context, token string, q VoiceQuery) ([]Voice, error) {
	var out struct {
		Voices []struct {
			VoiceID     string         `json:"voice_id"`
			Name        string         `json:"name"`
			Category    string         `json:"category"`
			Description string         `json:"description"`
			Labels      map[string]any `json:"labels"`
		} `json:"voices"`
	}
	if err := p.getJSON(ctx, token, "/v2/voices", q.values(), &out); err != nil {
		return nil, err
	}
	voices := make([]Voice, 0, len(out.Voices))
	for _, v := range out.Voices {
		label := func(k string) string {
			if s, ok := v.Labels[k].(string); ok {
				return s
			}
			return ""
		}
		desc := v.Description
		if strings.TrimSpace(desc) == "" {
			desc = label("descriptive")
		}
		voices = append(voices, Voice{
			Source:      "account",
			ID:          v.VoiceID,
			Name:        v.Name,
			Accent:      label("accent"),
			Age:         label("age"),
			Gender:      label("gender"),
			Description: desc,
			Category:    v.Category,
		})
	}
	return voices, nil
}

// LibraryVoices searches the shared voice library via GET /v1/shared-voices,
// filtered server-side by q.
func (p *ElevenLabsProvider) LibraryVoices(ctx context.Context, token string, q VoiceQuery) ([]Voice, error) {
	var out struct {
		Voices []struct {
			VoiceID     string `json:"voice_id"`
			Name        string `json:"name"`
			Accent      string `json:"accent"`
			Age         string `json:"age"`
			Gender      string `json:"gender"`
			Descriptive string `json:"descriptive"`
			Description string `json:"description"`
			Category    string `json:"category"`
		} `json:"voices"`
	}
	if err := p.getJSON(ctx, token, "/v1/shared-voices", q.values(), &out); err != nil {
		return nil, err
	}
	voices := make([]Voice, 0, len(out.Voices))
	for _, v := range out.Voices {
		desc := v.Description
		if strings.TrimSpace(desc) == "" {
			desc = v.Descriptive
		}
		voices = append(voices, Voice{
			Source:      "library",
			ID:          v.VoiceID,
			Name:        v.Name,
			Accent:      v.Accent,
			Age:         v.Age,
			Gender:      v.Gender,
			Description: desc,
			Category:    v.Category,
		})
	}
	return voices, nil
}

func (p *ElevenLabsProvider) getJSON(ctx context.Context, token, path string, query url.Values, out any) error {
	endpoint := p.base() + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	hreq.Header.Set("xi-api-key", token)
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("elevenlabs API %d: %s", resp.StatusCode, parseElevenLabsError(body).text(body))
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decode %s: %w", path, err)
	}
	return nil
}
