package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestStripAudioInPlace synthesizes a 1s clip with both a video and an audio
// stream, strips it, and asserts the audio is gone while the file survives.
// Skips cleanly when ffmpeg/ffprobe (or the needed encoders) are unavailable,
// so it never breaks CI on a host without media tooling.
func TestStripAudioInPlace(t *testing.T) {
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
		"-f", "lavfi", "-i", "testsrc=duration=1:size=128x128:rate=10",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=1",
		"-shortest", "-c:v", "libx264", "-c:a", "aac", in)
	if out, err := mk.CombinedOutput(); err != nil {
		t.Skipf("could not synthesize test video: %v: %s", err, strings.TrimSpace(string(out)))
	}

	if !hasAudioStream(ffprobe, in) {
		t.Fatal("synthesized input unexpectedly has no audio stream")
	}
	if err := stripAudioInPlace(ffmpeg, in); err != nil {
		t.Fatalf("stripAudioInPlace: %v", err)
	}
	if _, err := os.Stat(in); err != nil {
		t.Fatalf("output file missing after strip: %v", err)
	}
	if hasAudioStream(ffprobe, in) {
		t.Fatal("audio stream still present after strip")
	}
}

func hasAudioStream(ffprobe, path string) bool {
	out, _ := exec.Command(ffprobe, "-v", "error", "-select_streams", "a",
		"-show_entries", "stream=index", "-of", "csv=p=0", path).Output()
	return strings.TrimSpace(string(out)) != ""
}
