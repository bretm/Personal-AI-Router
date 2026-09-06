// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"fmt"
	"strings"

	"nvpair-tui/rpc"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
)

type authStatus struct {
	Initialized   bool   `json:"initialized"`
	Administrator bool   `json:"administrator"`
	ClusterID     string `json:"clusterId"`
	Revision      uint64 `json:"revision"`
}

type authClient struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Revoked bool   `json:"revoked"`
}

type authStatusMsg struct {
	status authStatus
	err    error
}

type authClientsMsg struct {
	clients []authClient
	err     error
}

type authMutationMsg struct {
	token string
	err   error
}

type credentialStatusMsg struct {
	configured bool
	source     string
	err        error
}

type authInputMode uint8

const (
	authInputNone authInputMode = iota
	authInputClientLabel
	authInputCredentialRef
	authInputCredentialValue
)

type authView struct {
	client               *rpc.Client
	status               authStatus
	clients              []authClient
	cursor               int
	input                textinput.Model
	inputMode            authInputMode
	credentialRef        string
	credentialConfigured bool
	credentialSource     string
	oneTimeToken         string
	message              string
	width, height        int
}

var (
	authUpKey        = key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("up/k", "up"))
	authDownKey      = key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("down/j", "down"))
	authBootstrapKey = key.NewBinding(key.WithKeys("b"), key.WithHelp("b", "initialize"))
	authNewKey       = key.NewBinding(key.WithKeys("n"), key.WithHelp("n", "new client"))
	authRevokeKey    = key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "revoke"))
	authRefKey       = key.NewBinding(key.WithKeys("e"), key.WithHelp("e", "credential ref"))
	authSetKey       = key.NewBinding(key.WithKeys("s"), key.WithHelp("s", "set credential"))
	authClearKey     = key.NewBinding(key.WithKeys("c"), key.WithHelp("c", "clear credential"))
	authDismissKey   = key.NewBinding(key.WithKeys("x"), key.WithHelp("x", "dismiss token"))
)

func newAuthView(client *rpc.Client) *authView {
	input := textinput.New()
	return &authView{
		client:           client,
		input:            input,
		credentialRef:    "engine.example.api_key",
		credentialSource: "missing",
	}
}

func (v *authView) Title() string { return "Auth" }

func (v *authView) Init() tea.Cmd { return tea.Batch(v.loadStatus(), v.inspectCredential()) }

func (v *authView) SetSize(width, height int) { v.width, v.height = width, height }

func (v *authView) CapturingInput() bool { return v.inputMode != authInputNone }

func (v *authView) loadStatus() tea.Cmd {
	return call(v.client, "auth:status", nil, func(msg *rpc.Message, err error) tea.Msg {
		var status authStatus
		if err == nil {
			err = decodeParams(msg.Result, &status)
		}
		return authStatusMsg{status: status, err: err}
	})
}

func (v *authView) loadClients() tea.Cmd {
	return call(v.client, "auth:list-clients", nil, func(msg *rpc.Message, err error) tea.Msg {
		var result struct {
			Clients []authClient `json:"clients"`
		}
		if err == nil {
			err = decodeParams(msg.Result, &result)
		}
		return authClientsMsg{clients: result.Clients, err: err}
	})
}

func (v *authView) inspectCredential() tea.Cmd {
	ref := v.credentialRef
	return call(v.client, "settings/get-engine-credential-status", map[string]string{"credential": ref}, func(msg *rpc.Message, err error) tea.Msg {
		var result struct {
			Configured bool   `json:"configured"`
			Source     string `json:"source"`
		}
		if err == nil {
			err = decodeParams(msg.Result, &result)
		}
		return credentialStatusMsg{configured: result.Configured, source: result.Source, err: err}
	})
}

func (v *authView) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case authStatusMsg:
		if msg.err != nil {
			v.message = "auth status failed: " + msg.err.Error()
			return nil
		}
		v.status = msg.status
		if v.status.Administrator {
			return v.loadClients()
		}
	case authClientsMsg:
		if msg.err != nil {
			v.message = "client list failed: " + msg.err.Error()
			return nil
		}
		v.clients = msg.clients
		if v.cursor >= len(v.clients) && v.cursor > 0 {
			v.cursor = len(v.clients) - 1
		}
	case authMutationMsg:
		if msg.err != nil {
			v.message = "operation failed: " + msg.err.Error()
			return nil
		}
		v.message = "saved"
		if msg.token != "" {
			v.oneTimeToken = msg.token
		}
		return tea.Batch(v.loadStatus(), v.inspectCredential())
	case credentialStatusMsg:
		if msg.err != nil {
			v.message = "credential status failed: " + msg.err.Error()
			return nil
		}
		v.credentialConfigured = msg.configured
		v.credentialSource = msg.source
	case tea.KeyMsg:
		return v.handleKey(msg)
	}
	return nil
}

func (v *authView) handleKey(msg tea.KeyMsg) tea.Cmd {
	if v.inputMode != authInputNone {
		switch msg.String() {
		case "enter":
			return v.submitInput()
		case "esc":
			v.stopInput()
			return nil
		}
		var cmd tea.Cmd
		v.input, cmd = v.input.Update(msg)
		return cmd
	}
	switch {
	case key.Matches(msg, authUpKey):
		if v.cursor > 0 {
			v.cursor--
		}
	case key.Matches(msg, authDownKey):
		if v.cursor+1 < len(v.clients) {
			v.cursor++
		}
	case key.Matches(msg, authBootstrapKey):
		return call(v.client, "auth:bootstrap-owner", nil, authMutationDecoder)
	case key.Matches(msg, authNewKey):
		if v.status.Administrator {
			return v.startInput(authInputClientLabel, "")
		}
	case key.Matches(msg, authRevokeKey):
		if v.status.Administrator && len(v.clients) > 0 && !v.clients[v.cursor].Revoked {
			id := v.clients[v.cursor].ID
			return call(v.client, "auth:revoke-client", map[string]string{"id": id}, authMutationDecoder)
		}
	case key.Matches(msg, authRefKey):
		return v.startInput(authInputCredentialRef, v.credentialRef)
	case key.Matches(msg, authSetKey):
		return v.startInput(authInputCredentialValue, "")
	case key.Matches(msg, authClearKey):
		return call(v.client, "settings/clear-engine-credential", map[string]string{"credential": v.credentialRef}, authMutationDecoder)
	case key.Matches(msg, authDismissKey):
		v.oneTimeToken = ""
	}
	return nil
}

func authMutationDecoder(msg *rpc.Message, err error) tea.Msg {
	var result struct {
		Token string `json:"token"`
	}
	if err == nil {
		err = decodeParams(msg.Result, &result)
	}
	return authMutationMsg{token: result.Token, err: err}
}

func (v *authView) startInput(mode authInputMode, value string) tea.Cmd {
	v.inputMode = mode
	v.input.SetValue(value)
	v.input.EchoMode = textinput.EchoNormal
	if mode == authInputCredentialValue {
		v.input.EchoMode = textinput.EchoPassword
	}
	v.input.Focus()
	return textinput.Blink
}

func (v *authView) stopInput() {
	v.inputMode = authInputNone
	v.input.SetValue("")
	v.input.Blur()
}

func (v *authView) submitInput() tea.Cmd {
	value := strings.TrimSpace(v.input.Value())
	mode := v.inputMode
	v.stopInput()
	if value == "" {
		return nil
	}
	switch mode {
	case authInputClientLabel:
		return call(v.client, "auth:create-client", map[string]string{"label": value}, authMutationDecoder)
	case authInputCredentialRef:
		v.credentialRef = value
		return v.inspectCredential()
	case authInputCredentialValue:
		params := map[string]string{"credential": v.credentialRef, "value": value}
		return call(v.client, "settings/set-engine-credential", params, authMutationDecoder)
	default:
		return nil
	}
}

func (v *authView) View() string {
	var b strings.Builder
	role := "not initialized"
	if v.status.Initialized {
		role = "validator"
	}
	if v.status.Administrator {
		role = "administrator"
	}
	fmt.Fprintf(&b, "PAIR client access: %s (revision %d)\n", role, v.status.Revision)
	if v.status.Administrator {
		for i, client := range v.clients {
			cursor := "  "
			if i == v.cursor {
				cursor = "> "
			}
			state := "active"
			if client.Revoked {
				state = "revoked"
			}
			fmt.Fprintf(&b, "%s%-24s %-8s %s\n", cursor, client.Label, state, client.ID)
		}
		if len(v.clients) == 0 {
			b.WriteString("  No clients yet.\n")
		}
	}
	if v.oneTimeToken != "" {
		b.WriteString("\nSave this token now; PAIR cannot show it again:\n")
		b.WriteString(v.oneTimeToken + "\n")
		b.WriteString("Press x after saving it.\n")
	}
	fmt.Fprintf(&b, "\nLocal engine credential\n  Reference: %s\n", v.credentialRef)
	state := "not configured"
	if v.credentialConfigured {
		state = "configured (" + v.credentialSource + ")"
	}
	b.WriteString("  Status: " + state + "\n")
	if v.inputMode != authInputNone {
		b.WriteString("\nValue: " + v.input.View() + "\n")
	}
	if v.message != "" {
		b.WriteString("\n" + footerStyle.Render(v.message))
	}
	return b.String()
}

func (v *authView) Help() []key.Binding {
	return []key.Binding{authUpKey, authDownKey, authBootstrapKey, authNewKey, authRevokeKey, authRefKey, authSetKey, authClearKey, authDismissKey}
}
