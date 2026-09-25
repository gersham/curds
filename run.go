package curds

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

// RunResult is the outcome of a generic Replicate passthrough run: the
// prediction id, its output (raw JSON, whatever shape the model returned), the
// URLs that output carried, and the upstream prediction JSON verbatim.
type RunResult struct {
	ID         string
	Output     json.RawMessage
	URLs       []string
	Prediction json.RawMessage
}

// Run creates one Replicate prediction from a caller-supplied input map and
// polls it to a terminal status. It applies no field mapping and downloads
// nothing, so it can drive models curds does not wrap first-class — that is
// what `curds run` is for. A failure carries the upstream error text.
func (p *ReplicateProvider) Run(ctx context.Context, req *Request, input map[string]any) (*RunResult, error) {
	pred, err := p.runPrediction(ctx, req, input)
	if err != nil {
		return nil, err
	}
	var urls []string
	all, err := extractOutputURLs(pred.Output)
	if err != nil {
		logDebug(req, "run.output_shape", "err", err.Error())
	}
	// Text models return plain strings (or arrays of streamed tokens) in the
	// same shape as file URLs; only real http(s) URLs are downloadable. Any
	// non-URL string means the output is text: print it, download nothing.
	for _, u := range all {
		if !strings.HasPrefix(u, "https://") && !strings.HasPrefix(u, "http://") {
			urls = nil
			break
		}
		urls = append(urls, u)
	}
	return &RunResult{ID: pred.ID, Output: pred.Output, URLs: urls, Prediction: pred.raw}, nil
}

// ParseRunInput converts `key=value` arguments into a Replicate prediction
// input map. A value starting with @ is a local file, uploaded as a data URL
// (http(s) and data URLs are passed through as-is). Anything else is decoded as
// JSON when it parses — numbers, booleans, arrays, objects, quoted strings —
// and otherwise sent as a plain string.
//
// Repeating a key collects the values into an array, so
// `ref=@a.png ref=@b.png` sends two references. Note that one value holding a
// comma-separated list is NOT split: pass JSON (`ref=["https://…","…"]`) or
// repeat the key.
func ParseRunInput(args []string) (map[string]any, error) {
	order := make([]string, 0, len(args))
	values := make(map[string][]any, len(args))
	for _, arg := range args {
		key, value, ok := strings.Cut(arg, "=")
		if !ok {
			return nil, fmt.Errorf("%q is not key=value", arg)
		}
		key = strings.TrimSpace(key)
		if key == "" {
			return nil, fmt.Errorf("%q has an empty key", arg)
		}
		parsed, err := parseRunValue(value)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", key, err)
		}
		if _, seen := values[key]; !seen {
			order = append(order, key)
		}
		values[key] = append(values[key], parsed)
	}
	input := make(map[string]any, len(order))
	for _, key := range order {
		vals := values[key]
		if len(vals) == 1 {
			input[key] = vals[0]
			continue
		}
		input[key] = vals
	}
	return input, nil
}

// parseRunValue turns one `key=value` value into the JSON shape Replicate
// expects.
func parseRunValue(value string) (any, error) {
	if rest, ok := strings.CutPrefix(value, "@"); ok {
		urls, err := encodeMediaAsDataURLs([]string{rest})
		if err != nil {
			return nil, err
		}
		return urls[0], nil
	}
	var decoded any
	if err := json.Unmarshal([]byte(value), &decoded); err == nil {
		return decoded, nil
	}
	return value, nil
}

// SchemaField is one input of a Replicate model's OpenAPI schema, as reported
// by `curds run -schema`.
type SchemaField struct {
	Name        string
	Type        string
	Default     string
	HasDefault  bool
	Enum        []string
	Minimum     *float64
	Maximum     *float64
	Description string
}

// String renders the field as one logfmt line: the name and type, then whatever
// constraints the schema actually declares.
func (f SchemaField) String() string {
	kv := []any{"name", f.Name, "type", f.Type}
	if f.HasDefault {
		kv = append(kv, "default", f.Default)
	}
	if len(f.Enum) > 0 {
		kv = append(kv, "enum", "["+strings.Join(f.Enum, ",")+"]")
	}
	if f.Minimum != nil {
		kv = append(kv, "min", formatSchemaNumber(*f.Minimum))
	}
	if f.Maximum != nil {
		kv = append(kv, "max", formatSchemaNumber(*f.Maximum))
	}
	if f.Description != "" {
		kv = append(kv, "description", f.Description)
	}
	return formatSchemaKV(kv)
}

// Schema fetches a model's input schema — the fields a prediction accepts — and
// returns them sorted by name. This is the document Replicate publishes on the
// model page (latest_version.openapi_schema); enums that the properties
// reference through allOf/$ref are resolved against components.schemas.
func (p *ReplicateProvider) Schema(ctx context.Context, token, model string) ([]SchemaField, error) {
	if strings.TrimSpace(model) == "" {
		return nil, errors.New("model is required")
	}
	if strings.Contains(model, ":") {
		return nil, fmt.Errorf("schema needs an official owner/name model (no :version pin), got %q", model)
	}
	endpoint := fmt.Sprintf("%s/models/%s", strings.TrimSuffix(p.base(), "/"), model)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		hreq.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := httpClientOrDefault(p.HTTPClient).Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("replicate API %d: %s", resp.StatusCode, truncate(string(body), 500))
	}
	return parseReplicateSchema(body)
}

// schemaNode is one node of the OpenAPI schema subtree Replicate publishes.
// Unknown keys are ignored; the ones curds prints are all here.
type schemaNode struct {
	Type        string                `json:"type"`
	Description string                `json:"description"`
	Default     json.RawMessage       `json:"default"`
	Enum        []json.RawMessage     `json:"enum"`
	Minimum     *float64              `json:"minimum"`
	Maximum     *float64              `json:"maximum"`
	Properties  map[string]schemaNode `json:"properties"`
	AllOf       []schemaRef           `json:"allOf"`
	AnyOf       []schemaRef           `json:"anyOf"`
	Ref         string                `json:"$ref"`
}

// schemaRef is an allOf/anyOf entry; only $ref entries carry anything curds
// needs.
type schemaRef struct {
	Ref string `json:"$ref"`
}

type replicateSchemaDoc struct {
	LatestVersion struct {
		OpenAPISchema struct {
			Components struct {
				Schemas map[string]json.RawMessage `json:"schemas"`
			} `json:"components"`
		} `json:"openapi_schema"`
	} `json:"latest_version"`
}

// parseReplicateSchema reads the Input properties out of a model document,
// resolving the $ref indirection Replicate uses for enum types.
func parseReplicateSchema(raw []byte) ([]SchemaField, error) {
	var doc replicateSchemaDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("decode model schema: %w", err)
	}
	schemas := doc.LatestVersion.OpenAPISchema.Components.Schemas
	inputRaw, ok := schemas["Input"]
	if !ok {
		return nil, errors.New("model document carries no Input schema")
	}
	var input schemaNode
	if err := json.Unmarshal(inputRaw, &input); err != nil {
		return nil, fmt.Errorf("decode input schema: %w", err)
	}
	fields := make([]SchemaField, 0, len(input.Properties))
	for name, prop := range input.Properties {
		fields = append(fields, schemaFieldFrom(name, resolveSchemaNode(prop, schemas, 0)))
	}
	sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
	return fields, nil
}

// resolveSchemaNode folds the schemas a node references (via $ref, allOf, or
// anyOf) into the node itself: whatever the reference declares and the node
// does not, the node inherits. depth caps the walk so a self-referential schema
// cannot spin.
func resolveSchemaNode(node schemaNode, schemas map[string]json.RawMessage, depth int) schemaNode {
	const maxDepth = 4
	if depth >= maxDepth {
		return node
	}
	refs := make([]string, 0, 2+len(node.AllOf))
	if node.Ref != "" {
		refs = append(refs, node.Ref)
	}
	for _, r := range node.AllOf {
		if r.Ref != "" {
			refs = append(refs, r.Ref)
		}
	}
	for _, r := range node.AnyOf {
		if r.Ref != "" {
			refs = append(refs, r.Ref)
		}
	}
	for _, ref := range refs {
		raw, ok := schemas[schemaRefName(ref)]
		if !ok {
			continue
		}
		var target schemaNode
		if err := json.Unmarshal(raw, &target); err != nil {
			continue
		}
		target = resolveSchemaNode(target, schemas, depth+1)
		mergeMissingSchemaFields(&node, target)
	}
	return node
}

// mergeMissingSchemaFields fills dst's unset fields from src.
func mergeMissingSchemaFields(dst *schemaNode, src schemaNode) {
	if dst.Type == "" {
		dst.Type = src.Type
	}
	if dst.Description == "" {
		dst.Description = src.Description
	}
	if len(dst.Default) == 0 {
		dst.Default = src.Default
	}
	if len(dst.Enum) == 0 {
		dst.Enum = src.Enum
	}
	if dst.Minimum == nil {
		dst.Minimum = src.Minimum
	}
	if dst.Maximum == nil {
		dst.Maximum = src.Maximum
	}
}

// schemaRefName is the last path segment of a JSON pointer like
// "#/components/schemas/mode".
func schemaRefName(ref string) string {
	if i := strings.LastIndex(ref, "/"); i >= 0 {
		return ref[i+1:]
	}
	return ref
}

// schemaFieldFrom renders one resolved property for display.
func schemaFieldFrom(name string, node schemaNode) SchemaField {
	field := SchemaField{
		Name:        name,
		Type:        node.Type,
		Enum:        schemaEnumStrings(node.Enum),
		Minimum:     node.Minimum,
		Maximum:     node.Maximum,
		Description: node.Description,
	}
	if def, ok := schemaScalar(node.Default); ok {
		field.Default = def
		field.HasDefault = true
	}
	return field
}

// schemaEnumStrings renders enum entries, which may be strings or numbers.
func schemaEnumStrings(raw []json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, entry := range raw {
		if v, ok := schemaScalar(entry); ok {
			out = append(out, v)
		}
	}
	return out
}

// schemaScalar renders a JSON scalar as text: strings as-is (unquoted), numbers
// and booleans in their JSON spelling. ok is false for absent values.
func schemaScalar(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return string(raw), true
	}
	switch t := v.(type) {
	case string:
		return t, true
	case bool:
		return strconv.FormatBool(t), true
	case float64:
		return formatSchemaNumber(t), true
	}
	b, err := json.Marshal(v)
	if err != nil {
		return string(raw), true
	}
	return string(b), true
}

func formatSchemaNumber(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// formatSchemaKV renders alternating key/value pairs as one logfmt line.
func formatSchemaKV(kv []any) string {
	parts := make([]string, 0, len(kv)/2)
	for i := 0; i+1 < len(kv); i += 2 {
		parts = append(parts, fmt.Sprintf("%v=%s", kv[i], logfmtQuote(fmt.Sprint(kv[i+1]))))
	}
	return strings.Join(parts, " ")
}
