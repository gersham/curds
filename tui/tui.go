// Package tui drives the curds interactive flow.
//
// The flow is split in two:
//
//  1. Initial token capture (one-shot via charmbracelet/huh) when no token
//     is available for the chosen provider. The user can save it back to
//     the config file from this screen.
//
//  2. Main loop (full bubbletea program with alt-screen) showing the
//     CURDS banner, a multi-line prompt, a spinner + scrolling log panel
//     during generation, the resulting paths, and a "Generate another?"
//     prompt that loops back with the prompt cleared.
package tui

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// Banner is the figlet rendered at the top of every TUI screen.
const banner = ` ██████╗██╗   ██╗██████╗ ██████╗ ███████╗
██╔════╝██║   ██║██╔══██╗██╔══██╗██╔════╝
██║     ██║   ██║██████╔╝██║  ██║███████╗
██║     ██║   ██║██╔══██╗██║  ██║╚════██║
╚██████╗╚██████╔╝██║  ██║██████╔╝███████║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝`

const subtitle = "to complement your fries and gravy"

var (
	bannerStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFC400")).Bold(true)
	subtitleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245")).Italic(true)
	hintStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	headingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("141")).Bold(true)
	logBoxStyle   = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("241")).
			Padding(0, 1)
	statusBarOK = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0B0B0B")).
			Background(lipgloss.Color("#A6E22E")).
			Bold(true).
			Padding(0, 1)
	statusBarErr = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(lipgloss.Color("#C0392B")).
			Bold(true).
			Padding(0, 1)
)

// Defaults seeds the TUI's initial values.
type Defaults struct {
	Provider     string
	Token        string // populated when caller already has one
	AspectRatio  string
	Quality      string
	NumImages    int
	OutputFormat string
	OutputPath   string // default file path; the TUI may override
	NeedToken    bool   // true when caller has no token yet
}

// TokenCaptured is returned by RunTokenCapture. It is empty when the user
// cancelled.
type TokenCaptured struct {
	Provider  string
	Token     string
	Save      bool
	Cancelled bool
}

// GenerateRequest is the contract from TUI -> caller-supplied generator.
type GenerateRequest struct {
	Prompt       string
	Provider     string
	Token        string
	AspectRatio  string
	Quality      string
	NumImages    int
	OutputFormat string
	OutputPath   string
}

// GenerateResult comes back to the TUI from the generator.
type GenerateResult struct {
	Paths    []string
	Duration time.Duration
	Err      error
}

// GenerateFn is the caller-provided hook that does the actual work.
// Implementations should write log lines (one event per line) to logsink
// as they progress; those lines flow into the on-screen log panel.
type GenerateFn func(ctx context.Context, req GenerateRequest, logsink io.Writer) GenerateResult

// ClearScreen blanks the terminal before any TUI rendering.
func ClearScreen() {
	// CSI sequences: erase entire screen + reset cursor to home.
	fmt.Print("\033[2J\033[H")
}

// RenderBanner prints the figlet + subtitle once. Used outside of bubbletea
// (e.g. before the huh-driven token capture).
func RenderBanner() {
	fmt.Println(bannerStyle.Render(banner))
	fmt.Println(subtitleStyle.Render(subtitle))
	fmt.Println()
}

// RunTokenCapture asks for provider + token. Returns Cancelled=true if the
// user aborted.
func RunTokenCapture(d Defaults) (*TokenCaptured, error) {
	out := &TokenCaptured{Provider: d.Provider}
	if out.Provider == "" {
		out.Provider = "openai"
	}
	form := huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("Provider").
				Description("Which provider should I use?").
				Options(
					huh.NewOption("OpenAI (direct)", "openai"),
					huh.NewOption("Replicate", "replicate"),
				).
				Value(&out.Provider),
			huh.NewInput().
				Title("API token").
				Description("Pasted tokens are masked. Leave blank to abort.").
				EchoMode(huh.EchoModePassword).
				Value(&out.Token),
			huh.NewConfirm().
				Title("Save token to ~/.config/curds/config.toml?").
				Affirmative("Yes, save").
				Negative("No, just this session").
				Value(&out.Save),
		),
	)
	if err := form.Run(); err != nil {
		out.Cancelled = true
		return out, nil
	}
	if strings.TrimSpace(out.Token) == "" {
		out.Cancelled = true
	}
	return out, nil
}

// ============================================================================
// Bubbletea main loop
// ============================================================================

type phase int

const (
	phasePrompt phase = iota
	phaseGenerating
	phaseResult
)

type model struct {
	phase    phase
	defaults Defaults
	gen      GenerateFn

	prompt   textarea.Model
	spin     spinner.Model
	logs     viewport.Model

	logBuffer []string
	logChan   chan string

	result *GenerateResult
	cancel context.CancelFunc

	width, height int
}

type logMsg string
type logsClosedMsg struct{}
type generateDoneMsg struct{ result GenerateResult }

// RunInteractive owns the main TUI loop. Call after RunTokenCapture (if
// needed). gen is invoked when the user submits the prompt; it should
// build a curds.Request, run it, save outputs, and return paths.
func RunInteractive(d Defaults, gen GenerateFn) error {
	ta := textarea.New()
	ta.Placeholder = "What should I generate?"
	ta.SetWidth(72)
	ta.SetHeight(6)
	ta.CharLimit = 8000
	ta.ShowLineNumbers = false
	ta.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFC400"))

	vp := viewport.New(72, 8)

	m := model{
		phase:    phasePrompt,
		defaults: d,
		gen:      gen,
		prompt:   ta,
		spin:     sp,
		logs:     vp,
	}

	prog := tea.NewProgram(m, tea.WithAltScreen())
	_, err := prog.Run()
	return err
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textarea.Blink,
		m.spin.Tick,
	)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		w := max(40, msg.Width-4)
		m.prompt.SetWidth(w)
		m.logs.Width = w
		m.logs.Height = max(6, msg.Height-len(strings.Split(banner, "\n"))-9)
		return m, nil

	case tea.KeyMsg:
		switch m.phase {
		case phasePrompt:
			switch msg.String() {
			case "ctrl+c", "esc":
				return m, tea.Quit
			case "enter":
				if strings.TrimSpace(m.prompt.Value()) == "" {
					return m, nil
				}
				return m.startGeneration()
			case "ctrl+j", "ctrl+enter", "alt+enter", "shift+enter":
				// Insert a literal newline. Different terminals use different
				// escape sequences for ctrl+enter; we accept all the common ones.
				m.prompt.InsertString("\n")
				return m, nil
			}
			var cmd tea.Cmd
			m.prompt, cmd = m.prompt.Update(msg)
			return m, cmd

		case phaseGenerating:
			if msg.String() == "ctrl+c" {
				if m.cancel != nil {
					m.cancel()
				}
				return m, tea.Quit
			}
			return m, nil

		case phaseResult:
			switch msg.String() {
			case "g", "G", "enter":
				return m.resetForAnother(), textarea.Blink
			case "q", "Q", "esc", "ctrl+c":
				return m, tea.Quit
			}
			return m, nil
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd

	case logMsg:
		// ANSI-aware soft wrap so colorized lines stay legible at any width.
		width := m.logs.Width - 2
		if width < 20 {
			width = 20
		}
		wrapped := ansi.Wrap(string(msg), width, " ")
		m.logBuffer = append(m.logBuffer, wrapped)
		// Tail the most recent N lines so very long runs don't OOM.
		const maxLines = 500
		if len(m.logBuffer) > maxLines {
			m.logBuffer = m.logBuffer[len(m.logBuffer)-maxLines:]
		}
		m.logs.SetContent(strings.Join(m.logBuffer, "\n"))
		m.logs.GotoBottom()
		return m, waitForLog(m.logChan)

	case logsClosedMsg:
		// Channel closed; no more log pumping cmds needed.
		return m, nil

	case generateDoneMsg:
		m.phase = phaseResult
		m.result = &msg.result
		return m, nil
	}
	return m, nil
}

func (m model) View() string {
	var b strings.Builder
	b.WriteString(bannerStyle.Render(banner))
	b.WriteString("\n")
	b.WriteString(subtitleStyle.Render(subtitle))
	b.WriteString("\n\n")

	switch m.phase {
	case phasePrompt:
		b.WriteString(headingStyle.Render("Prompt"))
		b.WriteString("\n")
		b.WriteString(m.prompt.View())
		b.WriteString("\n\n")
		b.WriteString(hintStyle.Render(fmt.Sprintf(
			"defaults: %s · %s · %s · n=%d   enter submit · ctrl+enter newline · ctrl+c quit",
			m.defaults.Provider, m.defaults.AspectRatio, m.defaults.Quality, m.defaults.NumImages,
		)))

	case phaseGenerating:
		b.WriteString(m.spin.View())
		b.WriteString(" ")
		b.WriteString(headingStyle.Render("Generating"))
		b.WriteString("\n\n")
		b.WriteString(headingStyle.Render("Logs"))
		b.WriteString("\n")
		b.WriteString(logBoxStyle.Render(m.logs.View()))
		b.WriteString("\n\n")
		b.WriteString(hintStyle.Render("ctrl+c to cancel"))

	case phaseResult:
		// Keep the log panel visible — the user wanted history preserved.
		b.WriteString(headingStyle.Render("Logs"))
		b.WriteString("\n")
		b.WriteString(logBoxStyle.Render(m.logs.View()))
		b.WriteString("\n\n")
		b.WriteString(m.renderStatusBar())
		b.WriteString("\n")
		b.WriteString(hintStyle.Render("[g] generate another · [q] quit"))
	}

	return b.String()
}

// renderStatusBar formats the post-generation status line.
//
// On success: "✓ Done in 8.4s · /path/to/file.webp" with a green pill.
// On failure: "✗ Failed: <error>" with a red pill.
// The bar fills the log-panel width so it visually anchors the bottom.
func (m model) renderStatusBar() string {
	if m.result == nil {
		return ""
	}
	width := m.logs.Width
	if width < 20 {
		width = 60
	}
	if m.result.Err != nil {
		return statusBarErr.Width(width).Render(fmt.Sprintf("✗ Failed: %s",
			ansi.Truncate(m.result.Err.Error(), width-12, "…"),
		))
	}
	paths := strings.Join(m.result.Paths, "  ")
	dur := m.result.Duration.Round(time.Millisecond).String()
	msg := fmt.Sprintf("✓ Done in %s · %s", dur, paths)
	return statusBarOK.Width(width).Render(ansi.Truncate(msg, width-2, "…"))
}

func (m model) startGeneration() (tea.Model, tea.Cmd) {
	m.phase = phaseGenerating
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.logChan = make(chan string, 256)

	req := GenerateRequest{
		Prompt:       strings.TrimSpace(m.prompt.Value()),
		Provider:     m.defaults.Provider,
		Token:        m.defaults.Token,
		AspectRatio:  m.defaults.AspectRatio,
		Quality:      m.defaults.Quality,
		NumImages:    m.defaults.NumImages,
		OutputFormat: m.defaults.OutputFormat,
		OutputPath:   m.defaults.OutputPath,
	}
	gen := m.gen
	logChan := m.logChan

	return m, tea.Batch(
		runGenerate(ctx, gen, req, logChan),
		waitForLog(logChan),
	)
}

func (m model) resetForAnother() model {
	m.prompt.Reset()
	m.prompt.Focus()
	m.logBuffer = nil
	m.logs.SetContent("")
	m.result = nil
	m.phase = phasePrompt
	return m
}

func runGenerate(ctx context.Context, gen GenerateFn, req GenerateRequest, logChan chan string) tea.Cmd {
	return func() tea.Msg {
		sink := newChanWriter(logChan)
		start := time.Now()
		res := gen(ctx, req, sink)
		res.Duration = time.Since(start)
		sink.Flush()
		close(logChan)
		return generateDoneMsg{result: res}
	}
}

func waitForLog(ch <-chan string) tea.Cmd {
	return func() tea.Msg {
		line, ok := <-ch
		if !ok {
			return logsClosedMsg{}
		}
		return logMsg(line)
	}
}

// chanWriter splits writes on '\n' and forwards each line to a channel.
// Drops lines if the channel is full so the UI never blocks the producer.
//
// Implements curds.ColorAware to opt into ANSI color from the library —
// otherwise IsTerminalWriter sees a plain io.Writer and emits uncolored
// logfmt.
type chanWriter struct {
	ch  chan<- string
	buf strings.Builder
}

func newChanWriter(ch chan<- string) *chanWriter { return &chanWriter{ch: ch} }

// WantsColor satisfies curds.ColorAware.
func (*chanWriter) WantsColor() bool { return true }

func (cw *chanWriter) Write(p []byte) (int, error) {
	cw.buf.Write(p)
	for {
		s := cw.buf.String()
		idx := strings.IndexByte(s, '\n')
		if idx < 0 {
			break
		}
		line := strings.TrimRight(s[:idx], "\r")
		cw.buf.Reset()
		cw.buf.WriteString(s[idx+1:])
		if line == "" {
			continue
		}
		select {
		case cw.ch <- line:
		default:
			// drop on overflow rather than block
		}
	}
	return len(p), nil
}

// Flush sends any buffered partial line.
func (cw *chanWriter) Flush() {
	if cw.buf.Len() == 0 {
		return
	}
	line := strings.TrimRight(cw.buf.String(), "\r\n")
	cw.buf.Reset()
	if line != "" {
		select {
		case cw.ch <- line:
		default:
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
