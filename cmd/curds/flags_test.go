package main

import (
	"errors"
	"flag"
	"strings"
	"testing"
)

// definedFlags mirrors the real flag surface closely enough to exercise the
// ambiguous short-form cases without standing up the whole CLI.
var definedFlags = []string{
	"prompt", "provider", "output", "output-format", "output-compression",
	"open", "number-of-images", "no-tui", "no-audio", "model", "mask",
	"moderation", "verbose", "video-duration", "video-resolution", "quality",
}

func TestSuggestFlagsAmbiguousShortForms(t *testing.T) {
	tests := []struct {
		typed string
		want  []string
	}{
		// -o and -n are the short forms curds deliberately does not define.
		// Every plausible expansion must be offered, shortest first.
		{"o", []string{"-open", "-output", "-output-format"}},
		{"n", []string{"-no-tui", "-no-audio", "-number-of-images"}},
		// -version does not exist; the closest real flags are the -v family.
		{"version", []string{"-verbose"}},
		// Prefixes resolve to the shortest exact continuation first.
		{"out", []string{"-output", "-output-format", "-output-compression"}},
		// "prom" also shares "pro" with -provider; -prompt must still lead.
		{"prom", []string{"-prompt", "-provider"}},
	}
	for _, tc := range tests {
		got := suggestFlags(tc.typed, definedFlags)
		if len(got) != len(tc.want) {
			t.Errorf("suggestFlags(%q) = %v, want %v", tc.typed, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("suggestFlags(%q) = %v, want %v", tc.typed, got, tc.want)
				break
			}
		}
	}
}

func TestSuggestFlagsCapsAndEmpty(t *testing.T) {
	if got := suggestFlags("", definedFlags); got != nil {
		t.Errorf("suggestFlags(\"\") = %v, want nil", got)
	}
	if got := suggestFlags("zqxjw", definedFlags); len(got) != 0 {
		t.Errorf("suggestFlags(\"zqxjw\") = %v, want no candidates", got)
	}
	if got := suggestFlags("o", definedFlags); len(got) > maxSuggestions {
		t.Errorf("suggestFlags(\"o\") returned %d candidates, want <= %d", len(got), maxSuggestions)
	}
}

// A one-edit typo on a short flag must not be "corrected" into a different real
// flag; only longer names get the wider tolerance.
func TestSuggestFlagsToleranceScalesWithLength(t *testing.T) {
	if got := suggestFlags("masc", definedFlags); len(got) == 0 || got[0] != "-mask" {
		t.Errorf("suggestFlags(\"masc\") = %v, want -mask first", got)
	}
	if got := suggestFlags("qualiti", definedFlags); len(got) == 0 || got[0] != "-quality" {
		t.Errorf("suggestFlags(\"qualiti\") = %v, want -quality first", got)
	}
}

func TestJoinOr(t *testing.T) {
	tests := []struct {
		in   []string
		want string
	}{
		{nil, ""},
		{[]string{"-a"}, "-a"},
		{[]string{"-a", "-b"}, "-a or -b"},
		{[]string{"-a", "-b", "-c"}, "-a, -b or -c"},
	}
	for _, tc := range tests {
		if got := joinOr(tc.in); got != tc.want {
			t.Errorf("joinOr(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"output", "output", 0},
		{"", "abc", 3},
		{"promt", "prompt", 1},
		{"mask", "model", 4},
	}
	for _, tc := range tests {
		if got := levenshtein(tc.a, tc.b); got != tc.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

// explainFlagError must name the typed flag and stay a single line — the whole
// point of the change is that the error is not buried under the usage dump.
func TestExplainFlagErrorUnknownFlag(t *testing.T) {
	fs := flag.NewFlagSet("curds", flag.ContinueOnError)
	fs.String("output", "", "")
	fs.Bool("open", false, "")
	old := flag.CommandLine
	flag.CommandLine = fs
	defer func() { flag.CommandLine = old }()

	got := explainFlagError(errors.New(undefinedFlagPrefix + "o")).Error()
	if !strings.Contains(got, "unknown flag -o") {
		t.Errorf("error %q does not name the typed flag", got)
	}
	if !strings.Contains(got, "-output") || !strings.Contains(got, "-open") {
		t.Errorf("error %q does not suggest the real flags", got)
	}
	if !strings.Contains(got, "curds -help") {
		t.Errorf("error %q does not point at the help", got)
	}
	if strings.Contains(got, "\n") {
		t.Errorf("error %q spans multiple lines", got)
	}
}

// Errors that are not "undefined flag" pass through untouched.
func TestExplainFlagErrorPassthrough(t *testing.T) {
	in := errors.New(`invalid value "x" for flag -number-of-images: parse error`)
	if got := explainFlagError(in); got != in {
		t.Errorf("explainFlagError rewrote an unrelated error: %v", got)
	}
}

func TestUsageErrorUnwraps(t *testing.T) {
	inner := errors.New("boom")
	err := error(&usageError{err: inner})
	if !errors.Is(err, inner) {
		t.Error("usageError does not unwrap to its cause")
	}
	var ue *usageError
	if !errors.As(err, &ue) {
		t.Error("usageError is not recoverable with errors.As")
	}
	if err.Error() != "boom" {
		t.Errorf("Error() = %q, want %q", err.Error(), "boom")
	}
}
