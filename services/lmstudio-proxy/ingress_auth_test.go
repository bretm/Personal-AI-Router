// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"net/http"
	"net/http/httptest"
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
