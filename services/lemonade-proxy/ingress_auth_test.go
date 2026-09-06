// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/engineauth"
	"nvpair-shared/meshauth"
)

type authMemoryStore map[string]string

func (s authMemoryStore) Get(key string) (string, error) {
	value, ok := s[key]
	if !ok {
		return "", errors.New("not found")
	}
	return value, nil
}

func (s authMemoryStore) Set(key, value string) error { s[key] = value; return nil }
func (s authMemoryStore) Remove(key string) error     { delete(s, key); return nil }

func lemonadeAuth(secret string) (engineauth.Config, engineauth.Store) {
	const ref = "engine.lemonade.api_key"
	return engineauth.Config{Scheme: engineauth.SchemeBearer, Credential: ref}, authMemoryStore{ref: secret}
}

func TestPrepareCandidateRequestInjectsOnlyForLocalLemonade(t *testing.T) {
	config, store := lemonadeAuth("destination-secret")
	p := &Proxy{engineAuth: config, credentialStore: store}

	local := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/v1/chat/completions", nil)
	local.Header.Set("Authorization", "Bearer caller-secret")
	local.Header.Set("Cookie", "session=caller")
	if err := p.prepareCandidateRequest(local, candidate{local: true}); err != nil {
		t.Fatal(err)
	}
	if got := local.Header.Get("Authorization"); got != "Bearer destination-secret" {
		t.Fatalf("local authorization = %q", got)
	}
	if got := local.Header.Get("Cookie"); got != "" {
		t.Fatalf("caller cookie leaked as %q", got)
	}

	remote := httptest.NewRequest(http.MethodPost, "https://peer/v1/chat/completions", nil)
	remote.Header.Set("Authorization", "Bearer caller-secret")
	if err := p.prepareCandidateRequest(remote, candidate{peerUUID: "peer"}); err != nil {
		t.Fatal(err)
	}
	if got := remote.Header.Get("Authorization"); got != "" {
		t.Fatalf("credential leaked to peer as %q", got)
	}

	manual := httptest.NewRequest(http.MethodPost, "http://manual/v1/chat/completions", nil)
	manual.Header.Set("Authorization", "Bearer caller-secret")
	if err := p.prepareCandidateRequest(manual, candidate{}); err != nil {
		t.Fatal(err)
	}
	if got := manual.Header.Get("Authorization"); got != "" {
		t.Fatalf("credential leaked to manual node as %q", got)
	}
}

func TestAuthorizeMeshClientRequiresAndStripsPairToken(t *testing.T) {
	registryStore := authMemoryStore{}
	registry := meshauth.New(t.TempDir())
	if _, err := registry.Bootstrap("cluster-a", registryStore); err != nil {
		t.Fatal(err)
	}
	_, token, err := registry.CreateClient("hermes", registryStore)
	if err != nil {
		t.Fatal(err)
	}
	p := &Proxy{meshAuth: registry}

	missing := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/v1/chat/completions", nil)
	missingRecorder := httptest.NewRecorder()
	if p.authorizeMeshClient(missingRecorder, missing) {
		t.Fatal("missing PAIR token was accepted")
	}
	if missingRecorder.Code != http.StatusUnauthorized {
		t.Fatalf("missing token status = %d", missingRecorder.Code)
	}

	valid := httptest.NewRequest(http.MethodPost, "http://127.0.0.1/v1/chat/completions", nil)
	valid.Header.Set("Authorization", "Bearer "+token)
	valid.Header.Set("Cookie", "session=caller")
	if !p.authorizeMeshClient(httptest.NewRecorder(), valid) {
		t.Fatal("valid PAIR token was rejected")
	}
	if got := valid.Header.Get("Authorization"); got != "" {
		t.Fatalf("PAIR token leaked as %q", got)
	}
	if got := valid.Header.Get("Cookie"); got != "" {
		t.Fatalf("caller cookie leaked as %q", got)
	}
}

func TestTerminalHopReportsMissingLemonadeCredential(t *testing.T) {
	p := &Proxy{engineAuth: engineauth.Config{
		Scheme: engineauth.SchemeBearer, Credential: "engine.lemonade.api_key",
	}}
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

func TestModelInventoryReportsMissingOrRejectedCredential(t *testing.T) {
	config := engineauth.Config{Scheme: engineauth.SchemeBearer, Credential: "engine.lemonade.api_key"}
	for _, tc := range []struct {
		name  string
		store engineauth.Store
	}{
		{name: "missing"},
		{name: "rejected", store: authMemoryStore{"engine.lemonade.api_key": "wrong-secret"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer correct-secret" {
					http.Error(w, "credential required", http.StatusUnauthorized)
					return
				}
				_, _ = w.Write([]byte(`{"data":[]}`))
			}))
			defer engine.Close()
			target, err := url.Parse(engine.URL)
			if err != nil {
				t.Fatal(err)
			}
			p := &Proxy{engineAuth: config, credentialStore: tc.store}
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/v1/models", nil)
			status, err := p.serveModelList(recorder, req, []candidate{{id: "local", url: target, local: true}})
			if status != http.StatusUnauthorized || err == nil {
				t.Fatalf("status = %d, err = %v", status, err)
			}
			if body := recorder.Body.String(); !strings.Contains(body, "engine-credential") {
				t.Fatalf("response = %q", body)
			}
		})
	}
}

func TestMTLSTerminalIngressUsesDestinationLemonadeCredential(t *testing.T) {
	const peerUUID = "peer-a"
	clusterDir := t.TempDir()
	clustertrusttest.Join(t, clusterDir, "cluster-a", "destination", peerUUID)
	peerCert := readPinnedCertificate(t, clusterDir, peerUUID)

	const destinationSecret = "destination-secret"
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+destinationSecret {
			http.Error(w, "credential required", http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer engine.Close()
	engineURL, err := url.Parse(engine.URL)
	if err != nil {
		t.Fatal(err)
	}
	config, store := lemonadeAuth(destinationSecret)
	p := &Proxy{mesh: clustertrust.Open(clusterDir), engineAuth: config, credentialStore: store}
	p.setLocalBackend(localBackend{Engine: "lemonade", Host: engineURL.Hostname(), Port: mustPort(t, engineURL), Healthy: true})

	req := httptest.NewRequest(http.MethodPost, "https://destination/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer caller-secret")
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{peerCert}}
	recorder := httptest.NewRecorder()
	p.handleClusterIngress(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("terminal response = %d: %s", recorder.Code, recorder.Body.String())
	}
}

func readPinnedCertificate(t *testing.T, clusterDir, peerUUID string) *x509.Certificate {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(clusterDir, "trusted", peerUUID+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var pin struct {
		CertPEM string `json:"certPem"`
	}
	if err := json.Unmarshal(body, &pin); err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode([]byte(pin.CertPEM))
	if block == nil {
		t.Fatal("pinned certificate is not PEM")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	return cert
}

func mustPort(t *testing.T, target *url.URL) int {
	t.Helper()
	port := target.Port()
	if port == "" {
		t.Fatal("URL has no port")
	}
	value, err := strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
