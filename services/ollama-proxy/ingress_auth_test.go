// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"nvpair-shared/engineauth"
	"nvpair-shared/meshauth"
)

type authMemoryStore map[string]string

func (s authMemoryStore) Get(key string) (string, error) {
	v, ok := s[key]
	if !ok {
		return "", errors.New("not found")
	}
	return v, nil
}
func (s authMemoryStore) Set(key, value string) error { s[key] = value; return nil }
func (s authMemoryStore) Remove(key string) error     { delete(s, key); return nil }

func TestPrepareCandidateRequestInjectsOnlyForLocalEngine(t *testing.T) {
	p := &Proxy{
		engineAuth:      engineauth.Config{Scheme: engineauth.SchemeBearer, Credential: "engine.test.api_key"},
		credentialStore: authMemoryStore{"engine.test.api_key": "engine-secret"},
	}
	local := httptest.NewRequest(http.MethodPost, "http://127.0.0.1", nil)
	local.Header.Set("Authorization", "Bearer client-token")
	if err := p.prepareCandidateRequest(local, candidate{local: true}); err != nil {
		t.Fatal(err)
	}
	if got := local.Header.Get("Authorization"); got != "Bearer engine-secret" {
		t.Fatalf("local authorization = %q", got)
	}

	remote := httptest.NewRequest(http.MethodPost, "http://127.0.0.1", nil)
	remote.Header.Set("Authorization", "Bearer client-token")
	remote.Header.Set("X-Engine-Key", "caller-secret")
	p.engineAuth = engineauth.Config{Scheme: engineauth.SchemeHeader, Credential: "engine.test.api_key", Header: "X-Engine-Key"}
	if err := p.prepareCandidateRequest(remote, candidate{peerUUID: "peer"}); err != nil {
		t.Fatal(err)
	}
	if got := remote.Header.Get("Authorization"); got != "" {
		t.Fatalf("remote authorization leaked as %q", got)
	}
	if got := remote.Header.Get("X-Engine-Key"); got != "" {
		t.Fatalf("remote engine header leaked as %q", got)
	}
}

func TestTerminalHopReportsMissingCredential(t *testing.T) {
	p := &Proxy{engineAuth: engineauth.Config{Scheme: engineauth.SchemeBearer, Credential: "engine.example.api_key"}}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/v1/chat/completions", nil)
	recorder := httptest.NewRecorder()
	p.reverseProxyToLocal(recorder, req, req.URL)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "engine-credential") || !strings.Contains(body, "not configured") {
		t.Fatalf("response = %q", body)
	}
}

func TestAuthorizeMeshClientRequiresAndStripsBearer(t *testing.T) {
	store := authMemoryStore{}
	registry := meshauth.New(t.TempDir())
	if _, err := registry.Bootstrap("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	_, token, err := registry.CreateClient("hermes", store)
	if err != nil {
		t.Fatal(err)
	}
	p := &Proxy{meshAuth: registry}

	missing := httptest.NewRequest(http.MethodPost, "http://127.0.0.1", nil)
	missingRecorder := httptest.NewRecorder()
	if p.authorizeMeshClient(missingRecorder, missing) || missingRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("missing bearer result = %v, status %d", missingRecorder.Code != 0, missingRecorder.Code)
	}

	valid := httptest.NewRequest(http.MethodPost, "http://127.0.0.1", nil)
	valid.Header.Set("Authorization", "Bearer "+token)
	valid.Header.Set("Cookie", "session=client")
	if !p.authorizeMeshClient(httptest.NewRecorder(), valid) {
		t.Fatal("valid mesh bearer was rejected")
	}
	if got := valid.Header.Get("Authorization"); got != "" {
		t.Fatalf("bearer leaked as %q", got)
	}
	if got := valid.Header.Get("Cookie"); got != "" {
		t.Fatalf("cookie leaked as %q", got)
	}
}

func TestTwoProxyMeshUsesPairTokenAndDestinationCredential(t *testing.T) {
	registryStore := authMemoryStore{}
	registry := meshauth.New(t.TempDir())
	if _, err := registry.Bootstrap("cluster-a", registryStore); err != nil {
		t.Fatal(err)
	}
	_, pairToken, err := registry.CreateClient("hermes", registryStore)
	if err != nil {
		t.Fatal(err)
	}

	const destinationSecret = "destination-engine-secret"
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+destinationSecret {
			http.Error(w, "wrong destination credential", http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer engine.Close()
	target, err := url.Parse(engine.URL)
	if err != nil {
		t.Fatal(err)
	}

	entry := &Proxy{
		meshAuth: registry,
		engineAuth: engineauth.Config{
			Scheme: engineauth.SchemeBearer, Credential: "engine.example.api_key",
		},
		credentialStore: authMemoryStore{"engine.example.api_key": "entry-engine-secret"},
	}
	destination := &Proxy{
		engineAuth: engineauth.Config{
			Scheme: engineauth.SchemeBearer, Credential: "engine.example.api_key",
		},
		credentialStore: authMemoryStore{"engine.example.api_key": destinationSecret},
	}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/api/generate", nil)
	req.Header.Set("Authorization", "Bearer "+pairToken)
	if !entry.authorizeMeshClient(httptest.NewRecorder(), req) {
		t.Fatal("entry proxy rejected the PAIR token")
	}
	if err := entry.prepareCandidateRequest(req, candidate{peerUUID: "destination"}); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Fatalf("entry proxy forwarded a credential: %q", got)
	}

	recorder := httptest.NewRecorder()
	destination.reverseProxyToLocal(recorder, req, target)
	if recorder.Code != http.StatusOK {
		t.Fatalf("destination response = %d: %s", recorder.Code, recorder.Body.String())
	}
}
