package curds

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strings"
)

// SupportsInlineImages reports whether the current terminal accepts the
// iTerm2-style OSC 1337 image protocol.
//
// Detection is best-effort by environment variable. Recognised:
//   - iTerm2          (TERM_PROGRAM=iTerm.app)
//   - WezTerm         (TERM_PROGRAM=WezTerm or WEZTERM_EXECUTABLE set)
//   - VS Code 1.85+   (TERM_PROGRAM=vscode)
//   - Ghostty         (TERM_PROGRAM=ghostty)
//   - Konsole recent  (KONSOLE_VERSION set)
//   - Tabby           (TERM_PROGRAM=Tabby)
//
// Kitty uses a separate (incompatible) graphics protocol and is *not*
// detected here.
func SupportsInlineImages() bool {
	switch strings.ToLower(os.Getenv("TERM_PROGRAM")) {
	case "iterm.app", "wezterm", "vscode", "ghostty", "tabby":
		return true
	}
	if os.Getenv("WEZTERM_EXECUTABLE") != "" {
		return true
	}
	if os.Getenv("KONSOLE_VERSION") != "" {
		return true
	}
	return false
}

// InsideTmux reports whether we appear to be running inside tmux.
func InsideTmux() bool {
	return os.Getenv("TMUX") != ""
}

// InlineImageOpts controls how an inline image is rendered.
type InlineImageOpts struct {
	Name           string // filename (used for "save" UI in iTerm2)
	WidthCells     int    // 0 = auto
	HeightCells    int    // 0 = auto
	PreserveAspect bool
}

// EncodeInlineImage builds the OSC 1337 escape sequence for displaying an
// image inline. When running inside tmux, the sequence is wrapped in
// tmux's passthrough envelope (which only works with `allow-passthrough on`).
//
// The returned string can be written directly to stdout/stderr in non-TUI
// mode, or embedded in a Bubble Tea View string in TUI mode.
func EncodeInlineImage(data []byte, opts InlineImageOpts) string {
	args := []string{"inline=1", fmt.Sprintf("size=%d", len(data))}
	if opts.Name != "" {
		args = append(args, "name="+base64.StdEncoding.EncodeToString([]byte(opts.Name)))
	}
	if opts.WidthCells > 0 {
		args = append(args, fmt.Sprintf("width=%d", opts.WidthCells))
	}
	if opts.HeightCells > 0 {
		args = append(args, fmt.Sprintf("height=%d", opts.HeightCells))
	}
	if opts.PreserveAspect {
		args = append(args, "preserveAspectRatio=1")
	}

	body := base64.StdEncoding.EncodeToString(data)
	seq := "\x1b]1337;File=" + strings.Join(args, ";") + ":" + body + "\x07"

	if InsideTmux() {
		// tmux passthrough: \ePtmux;<escaped sequence>\e\\
		// where each ESC inside <sequence> is doubled.
		escaped := strings.ReplaceAll(seq, "\x1b", "\x1b\x1b")
		seq = "\x1bPtmux;" + escaped + "\x1b\\"
	}
	return seq
}

// WriteInlineImage writes an inline image escape sequence followed by a
// newline. Convenience for non-TUI use.
func WriteInlineImage(w io.Writer, data []byte, opts InlineImageOpts) error {
	if _, err := io.WriteString(w, EncodeInlineImage(data, opts)); err != nil {
		return err
	}
	_, err := io.WriteString(w, "\n")
	return err
}
