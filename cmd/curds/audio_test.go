package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/gersham/curds"
)

// stubLookPath makes exec.LookPath report ffmpeg as missing (or present at a
// given path) for the duration of a test.
func stubLookPath(t *testing.T, path string, err error) {
	t.Helper()
	old := lookPath
	lookPath = func(string) (string, error) { return path, err }
	t.Cleanup(func() { lookPath = old })
}

func TestValidateAudioFlags(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		mut        func(o *cliOptions)
		wantErrSub string
	}{
		{"music with defaults", curds.MusicModel, func(o *cliOptions) {}, ""},
		{"music in range", curds.MusicModel, func(o *cliOptions) { o.duration = 300 }, ""},
		{"music too long", curds.MusicModel, func(o *cliOptions) { o.duration = 301 }, "5-300 seconds for -model music"},
		{"music too short", curds.MusicModel, func(o *cliOptions) { o.duration = 4.5 }, "5-300 seconds for -model music"},
		{"music rejects lyrics", curds.MusicModel, func(o *cliOptions) { o.lyrics = "la la" }, "-lyrics is only supported by -model music-vocal"},
		{"music-vocal lyrics", curds.MusicVocalModel, func(o *cliOptions) { o.lyrics = "[Verse]\nhi" }, ""},
		{"music-vocal sub-second", curds.MusicVocalModel, func(o *cliOptions) { o.duration = 0.5 }, "at least 1 second"},
		{"sfx in range", curds.SFXModel, func(o *cliOptions) { o.duration = 190 }, ""},
		{"sfx too long", curds.SFXModel, func(o *cliOptions) { o.duration = 191 }, "1-190 seconds for -model sfx"},
		{"sfx rejects lyrics", curds.SFXModel, func(o *cliOptions) { o.lyrics = "la" }, "-lyrics is only supported by -model music-vocal"},
		{"audio model accepts mp3", curds.MusicModel, func(o *cliOptions) { o.outputFormat = "mp3" }, ""},
		{"image model rejects mp3", "gpt-image-2.5", func(o *cliOptions) { o.outputFormat = "mp3" }, "needs an audio model"},
		{"image model rejects wav", "gpt-image-2.5", func(o *cliOptions) { o.outputFormat = "wav" }, "needs an audio model"},
		{"image model rejects lyrics", "gpt-image-2.5", func(o *cliOptions) { o.lyrics = "la" }, "-lyrics is only supported by -model music-vocal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts := &cliOptions{}
			tc.mut(opts)
			err := validateAudioFlags(tc.model, opts)
			if tc.wantErrSub == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("got %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestResolveLyrics(t *testing.T) {
	t.Run("inline text is left alone", func(t *testing.T) {
		opts := &cliOptions{lyrics: "hold the line"}
		if err := resolveLyrics(opts); err != nil {
			t.Fatalf("resolveLyrics: %v", err)
		}
		if opts.lyrics != "hold the line" {
			t.Errorf("lyrics: %q", opts.lyrics)
		}
	})
	t.Run("at-file is read and trimmed", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "song.txt")
		if err := os.WriteFile(path, []byte("\n[Verse]\nrain on the roof\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		opts := &cliOptions{lyrics: "@" + path}
		if err := resolveLyrics(opts); err != nil {
			t.Fatalf("resolveLyrics: %v", err)
		}
		if opts.lyrics != "[Verse]\nrain on the roof" {
			t.Errorf("lyrics: %q", opts.lyrics)
		}
	})
	t.Run("missing file fails loudly", func(t *testing.T) {
		opts := &cliOptions{lyrics: "@" + filepath.Join(t.TempDir(), "nope.txt")}
		if err := resolveLyrics(opts); err == nil || !strings.Contains(err.Error(), "read -lyrics file") {
			t.Fatalf("got %v, want a read error", err)
		}
	})
}

func TestAudioOutputFormat(t *testing.T) {
	cases := map[string]string{
		"":                "mp3",
		"/tmp/song.wav":   "wav",
		"/tmp/SONG.WAV":   "wav",
		"/tmp/song.mp3":   "mp3",
		"/tmp/song":       "mp3",
		"/tmp/song.webp":  "mp3",
		"/tmp/song.wav.t": "mp3",
	}
	for in, want := range cases {
		if got := audioOutputFormat(in); got != want {
			t.Errorf("audioOutputFormat(%q) = %q want %q", in, got, want)
		}
	}
}

func TestTrimAudioArgs(t *testing.T) {
	got := strings.Join(trimAudioArgs("/tmp/song.mp3", "/tmp/out.mp3", 60), " ")
	want := "-y -loglevel error -i /tmp/song.mp3 -t 60 -af afade=t=out:st=58:d=2 /tmp/out.mp3"
	if got != want {
		t.Errorf("trimAudioArgs = %q want %q", got, want)
	}
	// A trim shorter than the fade starts the fade at zero instead of going
	// negative (ffmpeg would otherwise refuse the filter).
	short := strings.Join(trimAudioArgs("/tmp/in.wav", "/tmp/out.wav", 1.5), " ")
	if !strings.Contains(short, "afade=t=out:st=0:d=2") {
		t.Errorf("short trim fade: %q", short)
	}
	if !strings.Contains(short, "-t 1.5") {
		t.Errorf("fractional seconds must survive: %q", short)
	}
}

// -duration only has a meaning for a music-vocal render; everything else skips
// the trim (and its ffmpeg lookup) entirely.
func TestMaybeTrimAudioSkips(t *testing.T) {
	stubLookPath(t, "", os.ErrNotExist)
	cases := []struct {
		name    string
		model   string
		seconds float64
		count   int
	}{
		{"other model", curds.MusicModel, 30, 1},
		{"no duration", curds.MusicVocalModel, 0, 1},
		{"no audio saved", curds.MusicVocalModel, 30, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var logs strings.Builder
			maybeTrimAudio(tc.model, tc.seconds, tc.count, []string{"/tmp/song.mp3"}, &logs)
			if logs.Len() != 0 {
				t.Errorf("nothing to do means no logs, got %q", logs.String())
			}
		})
	}
	// And with a real trim requested but no ffmpeg, it warns once and leaves
	// the file alone.
	var logs strings.Builder
	maybeTrimAudio(curds.MusicVocalModel, 30, 1, []string{"/tmp/song.mp3"}, &logs)
	if !strings.Contains(logs.String(), "event=audio.trim_skipped") {
		t.Errorf("expected a skip event, got %q", logs.String())
	}
	if !strings.Contains(logs.String(), "ffmpeg not found on PATH") {
		t.Errorf("the reason must name ffmpeg: %q", logs.String())
	}
}

// saveAudios keeps the requested container when the model agrees, and
// retargets the extension (with a warning) when it does not and ffmpeg is
// missing.
func TestSaveAudiosContainerHandling(t *testing.T) {
	t.Run("matching container", func(t *testing.T) {
		dir := t.TempDir()
		target := filepath.Join(dir, "song.mp3")
		opts := &cliOptions{outputPath: target, outputFormat: "mp3"}
		var printed string
		printed = captureStdout(t, func() {
			paths, err := saveAudios(opts, []curds.Audio{{Bytes: []byte("mp3-bytes"), Format: "mp3"}})
			if err != nil {
				t.Fatalf("saveAudios: %v", err)
			}
			if len(paths) != 1 || paths[0] != target {
				t.Fatalf("paths: %#v", paths)
			}
		})
		if !strings.Contains(printed, target) {
			t.Errorf("the saved path must go to stdout: %q", printed)
		}
		b, err := os.ReadFile(target)
		if err != nil || string(b) != "mp3-bytes" {
			t.Fatalf("file contents: %q err=%v", b, err)
		}
	})
	t.Run("mismatch without ffmpeg keeps the model container", func(t *testing.T) {
		stubLookPath(t, "", os.ErrNotExist)
		dir := t.TempDir()
		opts := &cliOptions{outputPath: filepath.Join(dir, "fx.mp3"), outputFormat: "mp3"}
		var paths []string
		captureStdout(t, func() {
			var err error
			paths, err = saveAudios(opts, []curds.Audio{{Bytes: []byte("wav-bytes"), Format: "wav"}})
			if err != nil {
				t.Fatalf("saveAudios: %v", err)
			}
		})
		want := filepath.Join(dir, "fx.wav")
		if len(paths) != 1 || paths[0] != want {
			t.Fatalf("paths: %#v want %#v", paths, want)
		}
		if _, err := os.Stat(want); err != nil {
			t.Fatalf("wav bytes must land under a .wav name: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "fx.mp3")); !os.IsNotExist(err) {
			t.Error("the mp3 name must not hold wav bytes")
		}
	})
}

// With ffmpeg available, a container mismatch is transcoded into the requested
// format instead of renaming the file.
func TestSaveAudiosTranscodesMismatch(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	src := filepath.Join(t.TempDir(), "tone.wav")
	if out, err := exec.Command(ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
		"-i", "sine=frequency=440:duration=2", src).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg generate: %v: %s", err, out)
	}
	wavBytes, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	target := filepath.Join(dir, "tone.mp3")
	opts := &cliOptions{outputPath: target, outputFormat: "mp3"}
	captureStdout(t, func() {
		paths, err := saveAudios(opts, []curds.Audio{{Bytes: wavBytes, Format: "wav"}})
		if err != nil {
			t.Fatalf("saveAudios: %v", err)
		}
		if len(paths) != 1 || paths[0] != target {
			t.Fatalf("paths: %#v want %#v", paths, target)
		}
	})
	info, err := os.Stat(target)
	if err != nil || info.Size() == 0 {
		t.Fatalf("transcoded file missing or empty: %v", err)
	}
	if info.Size() == int64(len(wavBytes)) {
		t.Error("an mp3 re-encode should not be byte-identical to the wav source")
	}
	// ffprobe confirms the container actually changed, not just the name.
	if ffprobe, err := exec.LookPath("ffprobe"); err == nil {
		out, err := exec.Command(ffprobe, "-v", "error", "-show_entries", "format=format_name",
			"-of", "default=nw=1:nk=1", target).CombinedOutput()
		if err != nil {
			t.Fatalf("ffprobe: %v: %s", err, out)
		}
		if !strings.Contains(string(out), "mp3") {
			t.Errorf("container: %q", out)
		}
	}
}

// trimAudioInPlace really shortens a render and applies the fade-out.
func TestTrimAudioInPlaceWithFFmpeg(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	path := filepath.Join(t.TempDir(), "song.wav")
	if out, err := exec.Command(ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
		"-i", "sine=frequency=330:duration=3", path).CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg generate: %v: %s", err, out)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	if err := trimAudioInPlace(ffmpeg, path, 1); err != nil {
		t.Fatalf("trimAudioInPlace: %v", err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() == 0 {
		t.Fatal("trimmed file is empty")
	}
	if after.Size() >= before.Size() {
		t.Errorf("a 1s trim of a 3s render must be smaller: %d -> %d", before.Size(), after.Size())
	}

	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		return
	}
	out, err := exec.Command(ffprobe, "-v", "error", "-show_entries", "format=duration",
		"-of", "default=nw=1:nk=1", path).CombinedOutput()
	if err != nil {
		t.Fatalf("ffprobe: %v: %s", err, out)
	}
	seconds, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		t.Fatalf("parse duration %q: %v", out, err)
	}
	if seconds > 1.5 {
		t.Errorf("trim left %fs of a 3s render, want ~1s", seconds)
	}
}

// The help is the contract users and agents read, so it must name the new
// section and every audio model key.
func TestHelpTextMentionsAudio(t *testing.T) {
	help := helpText()
	for _, want := range []string{
		"MUSIC & SOUND EFFECTS",
		"Music & sound effects (Replicate)",
		"-model music ",
		"-model music-vocal",
		"-model sfx",
		"-duration SECONDS",
		"-instrumental",
		"-lyrics TEXT|@FILE",
		"elevenlabs/music",
		"minimax/music-2.6",
		"stability-ai/stable-audio-2.5",
		"a 45-second instrumental cue with an exact length",
		"Song with lyrics read from a file",
		"8-second ambience sound effect",
	} {
		if !strings.Contains(help, want) {
			t.Errorf("help is missing %q", want)
		}
	}
}
