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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/gersham/curds"
)

// curdsEncode wraps the curds package helper so the file isn't
// re-imported throughout the TUI internals.
func curdsEncode(data []byte, path string, widthCells, heightCells int) string {
	w := widthCells
	if w < 10 {
		w = 0 // let the terminal auto-size if we don't have width yet
	}
	h := heightCells
	if h < 5 {
		h = 0
	}
	return curds.EncodeInlineImage(data, curds.InlineImageOpts{
		Name:           filepath.Base(path),
		WidthCells:     w,
		HeightCells:    h,
		PreserveAspect: true,
	})
}

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
	InlinePreview bool  // true to display the rendered images inline
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
	phaseSettings
)

// SettingsValues mirrors the config-file fields the user can edit through
// the in-TUI settings form.
type SettingsValues struct {
	Provider          string
	OutputDirectory   string
	OutputFormat      string
	OutputCompression int
	OpenAIToken       string
	ReplicateToken    string
	Quality           string
	AspectRatio       string
	Background        string
	Moderation        string
	NumberOfImages    int
}

// SaveSettingsFn persists the user's edited settings (e.g. by writing to
// ~/.config/curds/config.toml). Returning an error keeps the form open.
type SaveSettingsFn func(SettingsValues) error

type model struct {
	phase    phase
	defaults Defaults
	gen      GenerateFn

	prompt textarea.Model
	spin   spinner.Model
	logs   viewport.Model

	logBuffer []string
	logChan   chan string

	result *GenerateResult
	cancel context.CancelFunc

	// Inline image preview state, populated when phaseResult is reached and
	// defaults.InlinePreview is true.
	previews   []previewImage
	previewIdx int

	// Settings form state. settings holds the current persisted view; form
	// is non-nil only while phase == phaseSettings.
	settings     SettingsValues
	saveSettings SaveSettingsFn
	form         *huh.Form
	formValues   *settingsFormValues // bound to form fields

	width, height int
}

// settingsFormValues holds the live values bound to the huh form. Numeric
// fields are kept as strings because huh.NewInput binds to *string; we
// parse them back when applying.
type settingsFormValues struct {
	Provider          string
	OutputDirectory   string
	OutputFormat      string
	OutputCompression string
	OpenAIToken       string
	ReplicateToken    string
	Quality           string
	AspectRatio       string
	Background        string
	Moderation        string
	NumberOfImages    string
	Confirmed         bool // bound to the final "Save?" Confirm field
}

// previewImage holds a pre-encoded OSC 1337 sequence so the TUI doesn't
// re-base64 the file on every Bubble Tea render frame.
type previewImage struct {
	Path    string
	Encoded string
}

type logMsg string
type logsClosedMsg struct{}
type generateDoneMsg struct{ result GenerateResult }

// RunInteractive owns the main TUI loop. Call after RunTokenCapture (if
// needed). gen is invoked when the user submits the prompt; it should
// build a curds.Request, run it, save outputs, and return paths.
//
// settings + save power the in-TUI settings form (ctrl+s). settings
// supplies the current persisted view; save persists changes and is
// expected to update any state the caller relies on.
func RunInteractive(d Defaults, gen GenerateFn, settings SettingsValues, save SaveSettingsFn) error {
	m := newModel(d, gen, settings, save)
	prog := tea.NewProgram(m, tea.WithAltScreen())
	_, err := prog.Run()
	return err
}

// newModel is the constructor shared by RunInteractive and tests.
func newModel(d Defaults, gen GenerateFn, settings SettingsValues, save SaveSettingsFn) model {
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

	return model{
		phase:        phasePrompt,
		defaults:     d,
		gen:          gen,
		prompt:       ta,
		spin:         sp,
		logs:         vp,
		settings:     settings,
		saveSettings: save,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(
		textarea.Blink,
		m.spin.Tick,
	)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// While in the settings phase, route every message to the embedded
	// huh.Form. The form needs cursor-blink, init follow-up, and resize
	// messages — not just key presses — to reach an interactive state.
	// Without this, the form initialises but never advances on user input.
	if m.phase == phaseSettings && m.form != nil {
		if km, ok := msg.(tea.KeyMsg); ok && km.String() == "ctrl+c" {
			m.phase = phasePrompt
			m.form = nil
			m.formValues = nil
			return m, tea.Batch(tea.ClearScreen, textarea.Blink)
		}
		fm, cmd := m.form.Update(msg)
		if f, ok := fm.(*huh.Form); ok {
			m.form = f
		}
		switch m.form.State {
		case huh.StateCompleted:
			if m.formValues != nil && m.formValues.Confirmed {
				_ = m.applySettings()
			}
			m.phase = phasePrompt
			m.form = nil
			m.formValues = nil
			return m, tea.Batch(tea.ClearScreen, textarea.Blink)
		case huh.StateAborted:
			m.phase = phasePrompt
			m.form = nil
			m.formValues = nil
			return m, tea.Batch(tea.ClearScreen, textarea.Blink)
		}
		return m, cmd
	}

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
			case "ctrl+s":
				nm, formInit := m.openSettings()
				return nm, tea.Batch(tea.ClearScreen, formInit)
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

		case phaseSettings:
			if msg.String() == "ctrl+c" {
				m.phase = phasePrompt
				m.form = nil
				m.formValues = nil
				return m, tea.Batch(tea.ClearScreen, textarea.Blink)
			}
			fm, cmd := m.form.Update(msg)
			if f, ok := fm.(*huh.Form); ok {
				m.form = f
			}
			switch m.form.State {
			case huh.StateCompleted:
				// Only persist when the user picked "Save" on the final
				// Confirm field. Discard / form aborted both unwind without
				// touching the config file.
				if m.formValues != nil && m.formValues.Confirmed {
					_ = m.applySettings()
				}
				m.phase = phasePrompt
				m.form = nil
				m.formValues = nil
				return m, tea.Batch(tea.ClearScreen, textarea.Blink)
			case huh.StateAborted:
				m.phase = phasePrompt
				m.form = nil
				m.formValues = nil
				return m, tea.Batch(tea.ClearScreen, textarea.Blink)
			}
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
				return m.resetForAnother(), tea.Batch(tea.ClearScreen, textarea.Blink)
			case "tab", "right", "l":
				if len(m.previews) > 1 {
					m.previewIdx = (m.previewIdx + 1) % len(m.previews)
				}
				return m, tea.ClearScreen
			case "shift+tab", "left", "h":
				if len(m.previews) > 1 {
					m.previewIdx = (m.previewIdx - 1 + len(m.previews)) % len(m.previews)
				}
				return m, tea.ClearScreen
			case "d", "D":
				return m.deleteCurrentImage(), tea.ClearScreen
			case "q", "Q", "esc", "ctrl+c":
				return m, tea.Quit
			}
			return m, nil
		}

	case spinner.TickMsg:
		// Ignore spinner ticks once we're past the generating phase so the
		// view stops re-rendering on its own. Subsequent renders only fire
		// in response to user input, which lets the inline image stay put
		// instead of flashing.
		if m.phase != phaseGenerating {
			return m, nil
		}
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
		if m.defaults.InlinePreview && msg.result.Err == nil {
			m.previews = encodeAllPreviews(msg.result.Paths, m.logs.Width, m.logs.Height)
			m.previewIdx = 0
		}
		// Wipe the previous frame (Generating heading, spinner, log box) so
		// the inline image isn't drawn on top of leftover text cells.
		return m, tea.ClearScreen
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
			"defaults: %s · %s · %s · n=%d   enter submit · ctrl+enter newline · ctrl+s settings · ctrl+c quit",
			m.defaults.Provider, m.defaults.AspectRatio, m.defaults.Quality, m.defaults.NumImages,
		)))

	case phaseSettings:
		b.WriteString(headingStyle.Render("Settings"))
		b.WriteString("\n\n")
		if m.form != nil {
			b.WriteString(m.form.View())
		}

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
		// When inline preview is active, the image takes the slot the log
		// panel held during generation. Otherwise keep the log history
		// visible so the user can scroll back through events.
		if len(m.previews) > 0 {
			b.WriteString(m.renderPreview())
		} else {
			b.WriteString(headingStyle.Render("Logs"))
			b.WriteString("\n")
			b.WriteString(logBoxStyle.Render(m.logs.View()))
		}
		b.WriteString("\n\n")
		b.WriteString(m.renderStatusBar())
		b.WriteString("\n")
		hint := "[g] generate another · [q] quit"
		if len(m.previews) > 0 {
			if len(m.previews) > 1 {
				hint = "[tab] next · [d] delete · [g] generate another · [q] quit"
			} else {
				hint = "[d] delete · [g] generate another · [q] quit"
			}
		}
		b.WriteString(hintStyle.Render(hint))
	}

	return b.String()
}

// renderPreview returns the OSC 1337 escape sequence for the currently
// selected preview image, framed with a heading and an "image N of M"
// indicator. Bubble Tea writes this directly to the terminal; the
// terminal renders the image inline at the cursor position.
func (m model) renderPreview() string {
	if len(m.previews) == 0 {
		return ""
	}
	cur := m.previews[m.previewIdx]
	heading := headingStyle.Render("Preview")
	if len(m.previews) > 1 {
		heading += hintStyle.Render(fmt.Sprintf("  (image %d/%d)", m.previewIdx+1, len(m.previews)))
	}
	// Encoded already includes the BEL terminator. Trailing newline so
	// subsequent lines don't sit on top of the image's last row.
	return heading + "\n" + cur.Encoded + "\n"
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
		m.spin.Tick, // re-arm the spinner ticker for the generating phase
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
	m.previews = nil
	m.previewIdx = 0
	m.phase = phasePrompt
	return m
}

// deleteCurrentImage removes the currently-previewed image from disk and
// from the in-memory preview list. If the deletion fails, the file stays
// in the list and the user sees the original status bar (no error
// surfaced — the next regenerate or quit clears it).
func (m model) deleteCurrentImage() model {
	if len(m.previews) == 0 {
		return m
	}
	cur := m.previews[m.previewIdx]
	if err := os.Remove(cur.Path); err != nil {
		// Leave the entry in place so the user can retry; we don't have a
		// dedicated error region in the result view.
		return m
	}
	// Drop from preview list.
	m.previews = append(m.previews[:m.previewIdx], m.previews[m.previewIdx+1:]...)
	if m.previewIdx >= len(m.previews) && m.previewIdx > 0 {
		m.previewIdx--
	}
	// Drop from the result's saved-paths list so the status bar mirrors
	// reality.
	if m.result != nil {
		for i, p := range m.result.Paths {
			if p == cur.Path {
				m.result.Paths = append(m.result.Paths[:i], m.result.Paths[i+1:]...)
				break
			}
		}
	}
	return m
}

// openSettings populates a fresh huh.Form bound to the current settings
// and switches to phaseSettings. Returns the form's Init cmd so the
// caller can route it to bubbletea — without that, the form never gets
// its initial focus + blink + first-group activation.
func (m model) openSettings() (model, tea.Cmd) {
	values := &settingsFormValues{
		Provider:          m.settings.Provider,
		OutputDirectory:   m.settings.OutputDirectory,
		OutputFormat:      m.settings.OutputFormat,
		OutputCompression: fmt.Sprintf("%d", m.settings.OutputCompression),
		OpenAIToken:       m.settings.OpenAIToken,
		ReplicateToken:    m.settings.ReplicateToken,
		Quality:           m.settings.Quality,
		AspectRatio:       m.settings.AspectRatio,
		Background:        m.settings.Background,
		Moderation:        m.settings.Moderation,
		NumberOfImages:    fmt.Sprintf("%d", m.settings.NumberOfImages),
		// Default the final Save/Discard Confirm to "Save" so a user who
		// hits enter all the way through ends up persisting their changes.
		Confirmed: true,
	}
	m.formValues = values
	m.form = buildSettingsForm(values)
	m.phase = phaseSettings
	return m, m.form.Init()
}

// applySettings parses the form values back into SettingsValues, calls the
// user-provided save callback, and refreshes the TUI's prompt-screen
// defaults so the change takes effect immediately.
func (m *model) applySettings() error {
	if m.formValues == nil {
		return nil
	}
	v := m.formValues
	out := SettingsValues{
		Provider:        v.Provider,
		OutputDirectory: v.OutputDirectory,
		OutputFormat:    v.OutputFormat,
		OpenAIToken:     v.OpenAIToken,
		ReplicateToken:  v.ReplicateToken,
		Quality:         v.Quality,
		AspectRatio:     v.AspectRatio,
		Background:      v.Background,
		Moderation:      v.Moderation,
	}
	if n, err := strconv.Atoi(strings.TrimSpace(v.OutputCompression)); err == nil && n >= 0 && n <= 100 {
		out.OutputCompression = n
	} else {
		out.OutputCompression = m.settings.OutputCompression
	}
	if n, err := strconv.Atoi(strings.TrimSpace(v.NumberOfImages)); err == nil && n >= 1 && n <= 10 {
		out.NumberOfImages = n
	} else {
		out.NumberOfImages = m.settings.NumberOfImages
	}

	m.settings = out
	if m.saveSettings != nil {
		if err := m.saveSettings(out); err != nil {
			return err
		}
	}

	// Refresh prompt-screen defaults so the change is visible immediately.
	m.defaults.Provider = out.Provider
	m.defaults.AspectRatio = out.AspectRatio
	m.defaults.Quality = out.Quality
	m.defaults.NumImages = out.NumberOfImages
	m.defaults.OutputFormat = out.OutputFormat
	return nil
}

func buildSettingsForm(v *settingsFormValues) *huh.Form {
	aspectOpts := []huh.Option[string]{
		huh.NewOption("1:1 square", "1:1"),
		huh.NewOption("3:2 landscape", "3:2"),
		huh.NewOption("2:3 portrait", "2:3"),
		huh.NewOption("4:3 landscape", "4:3"),
		huh.NewOption("3:4 portrait", "3:4"),
		huh.NewOption("16:9 ~1080p+ landscape", "16:9"),
		huh.NewOption("9:16 ~1080p+ portrait", "9:16"),
		huh.NewOption("21:9 ultrawide", "21:9"),
		huh.NewOption("9:21 vertical ultra", "9:21"),
		huh.NewOption("2:1 landscape", "2:1"),
		huh.NewOption("1:2 portrait", "1:2"),
		huh.NewOption("16:9-4k landscape", "16:9-4k"),
		huh.NewOption("9:16-4k portrait", "9:16-4k"),
	}
	numImagesOpts := []huh.Option[string]{}
	for _, n := range []string{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"} {
		numImagesOpts = append(numImagesOpts, huh.NewOption(n, n))
	}

	// Single-group form so tab walks the whole list and enter on the last
	// field submits. Multi-group forms confused some users because tabbing
	// past the last field doesn't visibly advance to the next group.
	return huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("Provider").
				Description("Which backend curds should call by default.").
				Options(
					huh.NewOption("Auto-detect from tokens", ""),
					huh.NewOption("OpenAI direct (recommended)", "openai"),
					huh.NewOption("Replicate", "replicate"),
				).
				Value(&v.Provider),
			huh.NewSelect[string]().
				Title("Aspect ratio").
				Options(aspectOpts...).
				Value(&v.AspectRatio),
			huh.NewSelect[string]().
				Title("Quality").
				Options(
					huh.NewOption("auto (model picks)", "auto"),
					huh.NewOption("low (fast / cheap)", "low"),
					huh.NewOption("medium", "medium"),
					huh.NewOption("high", "high"),
				).
				Value(&v.Quality),
			huh.NewSelect[string]().
				Title("Number of images").
				Options(numImagesOpts...).
				Value(&v.NumberOfImages),
			huh.NewSelect[string]().
				Title("Output format").
				Options(
					huh.NewOption("webp (smallest)", "webp"),
					huh.NewOption("png (lossless)", "png"),
					huh.NewOption("jpeg", "jpeg"),
				).
				Value(&v.OutputFormat),
			huh.NewInput().
				Title("Output directory").
				Description("Where rendered images land. Tilde is expanded.").
				Value(&v.OutputDirectory),
			huh.NewInput().
				Title("Output compression (0-100)").
				Description("Applies to webp / jpeg via OpenAI.").
				Validate(validateInt(0, 100)).
				Value(&v.OutputCompression),
			huh.NewSelect[string]().
				Title("Background").
				Options(
					huh.NewOption("auto", "auto"),
					huh.NewOption("opaque", "opaque"),
				).
				Value(&v.Background),
			huh.NewSelect[string]().
				Title("Moderation").
				Options(
					huh.NewOption("auto (standard)", "auto"),
					huh.NewOption("low (less restrictive)", "low"),
				).
				Value(&v.Moderation),
			huh.NewInput().
				Title("OpenAI API token").
				Description("Stored in ~/.config/curds/config.toml. Leave blank to keep using .env / env.").
				EchoMode(huh.EchoModePassword).
				Value(&v.OpenAIToken),
			huh.NewInput().
				Title("Replicate API token").
				EchoMode(huh.EchoModePassword).
				Value(&v.ReplicateToken),
			huh.NewConfirm().
				Title("Save settings?").
				Description("Writes to ~/.config/curds/config.toml and applies immediately.").
				Affirmative("Save").
				Negative("Discard").
				Value(&v.Confirmed),
		),
	).WithShowHelp(true)
}

func validateInt(min, max int) func(string) error {
	return func(s string) error {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return fmt.Errorf("must be a number")
		}
		if n < min || n > max {
			return fmt.Errorf("must be between %d and %d", min, max)
		}
		return nil
	}
}

// encodeAllPreviews reads each path from disk and pre-encodes the OSC 1337
// sequence sized to fit the supplied cell dimensions. Errors per file are
// silently skipped so a single bad path doesn't kill the whole result view.
func encodeAllPreviews(paths []string, widthCells, heightCells int) []previewImage {
	out := make([]previewImage, 0, len(paths))
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		seq := curdsEncode(data, p, widthCells, heightCells)
		out = append(out, previewImage{Path: p, Encoded: seq})
	}
	return out
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
