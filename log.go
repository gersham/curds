package curds

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// Lipgloss styles for terminal output. ANSI codes are written only when the
// destination writer is a TTY; piped/redirected output stays plain logfmt
// for machine parsing.
var (
	styleTime  = lipgloss.NewStyle().Foreground(lipgloss.Color("242"))
	styleInfo  = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	styleError = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true)
	styleDebug = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	styleEvent = lipgloss.NewStyle().Foreground(lipgloss.Color("141")).Bold(true)
	styleKey   = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
)

// IsTerminalWriter reports whether w is a TTY (so it accepts ANSI colors).
// Non-*os.File writers are treated as not-a-terminal.
func IsTerminalWriter(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	stat, err := f.Stat()
	if err != nil || stat == nil {
		return false
	}
	return (stat.Mode() & os.ModeCharDevice) != 0
}

// FormatLogLine renders a logfmt event line. When color is true ANSI codes
// are inlined for use on a TTY; otherwise output stays plain logfmt for
// log collectors.
func FormatLogLine(level, event string, kv []any, color bool) string {
	ts := time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
	var b strings.Builder

	if color {
		b.WriteString(styleTime.Render("ts=" + ts))
		b.WriteString(" ")
		b.WriteString(levelStyle(level).Render("level=" + level))
		b.WriteString(" ")
		b.WriteString(styleEvent.Render("event=" + event))
		for i := 0; i+1 < len(kv); i += 2 {
			k, ok := kv[i].(string)
			if !ok {
				continue
			}
			v := fmt.Sprint(kv[i+1])
			b.WriteString(" ")
			b.WriteString(styleKey.Render(k + "="))
			b.WriteString(logfmtQuote(v))
		}
	} else {
		fmt.Fprintf(&b, "ts=%s level=%s event=%s", ts, level, event)
		for i := 0; i+1 < len(kv); i += 2 {
			k, ok := kv[i].(string)
			if !ok {
				continue
			}
			v := fmt.Sprint(kv[i+1])
			fmt.Fprintf(&b, " %s=%s", k, logfmtQuote(v))
		}
	}
	b.WriteByte('\n')
	return b.String()
}

func levelStyle(level string) lipgloss.Style {
	switch level {
	case "error":
		return styleError
	case "debug":
		return styleDebug
	default:
		return styleInfo
	}
}

func logEvent(w io.Writer, level, event string, kv ...any) {
	if w == nil {
		return
	}
	fmt.Fprint(w, FormatLogLine(level, event, kv, IsTerminalWriter(w)))
}

func logDebug(req *Request, event string, kv ...any) {
	if req == nil || !req.Verbose {
		return
	}
	logEvent(req.Logger, "debug", event, kv...)
}

func logInfo(req *Request, event string, kv ...any) {
	logEvent(req.Logger, "info", event, kv...)
}

func logError(req *Request, event string, kv ...any) {
	logEvent(req.Logger, "error", event, kv...)
}

func logfmtQuote(s string) string {
	if s == "" {
		return `""`
	}
	if strings.ContainsAny(s, ` "=\n\r\t`) {
		return strconv.Quote(s)
	}
	return s
}

// truncate keeps log values from blowing up when prompts/bodies are large.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "...(truncated)"
}
