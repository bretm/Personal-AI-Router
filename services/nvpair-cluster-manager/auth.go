// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"strings"
)

type authClientParams struct {
	Label string `json:"label"`
	ID    string `json:"id"`
}

func (m *Manager) clusterMemberCount() int {
	m.memMu.Lock()
	defer m.memMu.Unlock()
	count := 0
	for _, member := range m.members {
		if member.State == stateMember {
			count++
		}
	}
	return count
}

// Auth administration is deliberately local: a paired node can route work but
// cannot ask another node to use its owner key over the mesh.
func (m *Manager) handleAuthBootstrap(msg *Message) {
	clusterID, _ := m.clusterIdentity()
	if clusterID == "" {
		m.codec.RespondError(msg.ID, codePrecondition, "initialize mesh authentication: create a cluster first")
		return
	}
	if !m.meshAuth.Required() && m.clusterMemberCount() > 1 {
		m.codec.RespondError(msg.ID, codePrecondition, "initialize mesh authentication: only the founding node may initialize authentication before another node joins")
		return
	}
	doc, err := m.meshAuth.Bootstrap(clusterID, m.credentialStore)
	if err != nil {
		m.codec.RespondError(msg.ID, codePrecondition, "initialize mesh authentication: "+err.Error())
		return
	}
	m.codec.Respond(msg.ID, map[string]any{"clusterId": doc.ClusterID, "revision": doc.Revision})
}

func (m *Manager) handleAuthStatus(msg *Message) {
	status, err := m.meshAuth.Status(m.credentialStore)
	if err != nil {
		m.codec.RespondError(msg.ID, codePrecondition, "inspect mesh authentication: "+err.Error())
		return
	}
	clusterID, _ := m.clusterIdentity()
	if status.Initialized && status.ClusterID != clusterID {
		m.codec.RespondError(msg.ID, codePrecondition, "mesh authentication belongs to another cluster")
		return
	}
	m.codec.Respond(msg.ID, status)
}

func (m *Manager) handleAuthCreateClient(msg *Message) {
	var p authClientParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || strings.TrimSpace(p.Label) == "" {
		m.codec.RespondError(msg.ID, codeInvalidParams, `invalid params: expected {"label":"..."}`)
		return
	}
	client, token, err := m.meshAuth.CreateClient(strings.TrimSpace(p.Label), m.credentialStore)
	if err != nil {
		m.codec.RespondError(msg.ID, codePrecondition, "create mesh client: "+err.Error())
		return
	}
	// The raw token is returned exactly once to the local administrator. The
	// registry retains only its hash and list/revoke responses never reveal it.
	m.codec.Respond(msg.ID, map[string]any{"id": client.ID, "label": client.Label, "token": token})
}

func (m *Manager) handleAuthListClients(msg *Message) {
	status, statusErr := m.meshAuth.Status(m.credentialStore)
	if statusErr != nil || !status.Administrator {
		m.codec.RespondError(msg.ID, codePrecondition, "list mesh clients: this node is not the mesh administrator")
		return
	}
	doc, err := m.meshAuth.Export()
	if err != nil {
		m.codec.RespondError(msg.ID, codePrecondition, "mesh authentication is not initialized")
		return
	}
	type clientStatus struct {
		ID      string `json:"id"`
		Label   string `json:"label"`
		Revoked bool   `json:"revoked"`
	}
	out := make([]clientStatus, 0, len(doc.Clients))
	for _, c := range doc.Clients {
		out = append(out, clientStatus{ID: c.ID, Label: c.Label, Revoked: c.Revoked})
	}
	m.codec.Respond(msg.ID, map[string]any{"revision": doc.Revision, "clients": out})
}

func (m *Manager) handleAuthRevokeClient(msg *Message) {
	var p authClientParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || p.ID == "" {
		m.codec.RespondError(msg.ID, codeInvalidParams, `invalid params: expected {"id":"..."}`)
		return
	}
	if err := m.meshAuth.RevokeClient(p.ID, m.credentialStore); err != nil {
		m.codec.RespondError(msg.ID, codePrecondition, "revoke mesh client: "+err.Error())
		return
	}
	m.codec.Respond(msg.ID, map[string]bool{"ok": true})
}
