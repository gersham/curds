package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestCropCaptionsInPlace synthesizes a 1280x720 clip, crops it, and asserts the
// frame is the top 74% (rounded to even edges) with the audio track intact.
// Skips cleanly when ffmpeg/ffprobe are unavailable.
func TestCropCaptionsInPlace(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not installed")
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Skip("ffprobe not installed")
	}

	dir := t.TempDir()
	in := filepath.Join(dir, "v.mp4")
	mk := exec.Command(ffmpeg, "-y", "-loglevel", "error",
		"-f", "lavfi", "-i", "testsrc=duration=1:size=1280x720:rate=10",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=1",
		"-shortest", "-c:v", "libx264", "-c:a", "aac", in)
	if out, err := mk.CombinedOutput(); err != nil {
		t.Skipf("could not synthesize test video: %v: %s", err, strings.TrimSpace(string(out)))
	}

	if err := cropCaptionsInPlace(ffmpeg, in); err != nil {
		t.Fatalf("cropCaptionsInPlace: %v", err)
	}
	if _, err := os.Stat(in); err != nil {
		t.Fatalf("output file missing after crop: %v", err)
	}
	if got := probe(ffprobe, in, "v", "width,height"); got != "946,532" {
		t.Errorf("cropped frame = %q, want 946,532 (74%% of 1280x720, even edges)", got)
	}
	if !hasAudioStream(ffprobe, in) {
		t.Error("crop must copy the audio track")
	}
}

// probe returns the requested stream fields as csv for a stream type ("v"/"a").
func probe(ffprobe, path, stream, entries string) string {
	out, _ := exec.Command(ffprobe, "-v", "error", "-select_streams", stream,
		"-show_entries", "stream="+entries, "-of", "csv=p=0", path).Output()
	return strings.TrimSpace(string(out))
}
