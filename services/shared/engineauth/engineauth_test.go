// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engineauth

import (
	"errors"
	"net/http/httptest"
	"testing"
)

type memoryStore map[string]string

func (s memoryStore) Get(key string) (string, error) {
	v, ok := s[key]
	if !ok {
		return "", errors.New("not found")
	}
	return v, nil
}
func (s memoryStore) Set(key, value string) error { s[key] = value; return nil }
func (s memoryStore) Remove(key string) error     { delete(s, key); return nil }

func TestDeferredStoreDoesNotOpenUntilFirstOperation(t *testing.T) {
	opens := 0
	store := newDeferredStore(func() (Store, error) {
		opens++
		return memoryStore{"engine.example.api_key": "stored-secret"}, nil
	})
	if opens != 0 {
		t.Fatalf("constructing deferred store opened backend %d time(s)", opens)
	}
	if _, err := store.Get("engine.example.api_key"); err != nil {
		t.Fatal(err)
	}
	if err := store.Set("engine.example.api_key", "replacement"); err != nil {
		t.Fatal(err)
	}
	if err := store.Remove("engine.example.api_key"); err != nil {
		t.Fatal(err)
	}
	if opens != 1 {
		t.Fatalf("deferred store opened backend %d time(s), want 1", opens)
	}
}

func TestNativeStoreServiceNameIsStableIdentifier(t *testing.T) {
	if !environmentRE.MatchString(ServiceName) {
		t.Fatalf("native store service name %q is not a stable path identifier", ServiceName)
	}
}

func TestApplyUsesEnvironmentBeforeStore(t *testing.T) {
	t.Setenv("ENGINE_TEST_TOKEN", "environment-secret")
	c := Config{Scheme: SchemeBearer, Credential: "engine.test.api_key", Environment: "ENGINE_TEST_TOKEN"}
	req := httptest.NewRequest("GET", "http://127.0.0.1", nil)
	status, err := Apply(req, c, memoryStore{"engine.test.api_key": "stored-secret"})
	if err != nil {
		t.Fatal(err)
	}
	if !status.Configured || status.Source != SourceEnv {
		t.Fatalf("status = %+v", status)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer environment-secret" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestApplyUsesConfiguredHeader(t *testing.T) {
	c := Config{Scheme: SchemeHeader, Credential: "engine.example.api_key", Header: "X-Engine-Key"}
	req := httptest.NewRequest("GET", "http://127.0.0.1", nil)
	status, err := Apply(req, c, memoryStore{"engine.example.api_key": "stored-secret"})
	if err != nil {
		t.Fatal(err)
	}
	if !status.Configured || status.Source != SourceStore {
		t.Fatalf("status = %+v", status)
	}
	if got := req.Header.Get("X-Engine-Key"); got != "stored-secret" {
		t.Fatalf("engine header = %q", got)
	}
}

func TestApplyReportsMissingCredential(t *testing.T) {
	c := Config{Scheme: SchemeBearer, Credential: "engine.example.api_key"}
	req := httptest.NewRequest("GET", "http://127.0.0.1", nil)
	status, err := Apply(req, c, nil)
	if err == nil || status.Configured || status.Source != SourceMissing {
		t.Fatalf("Apply status = %+v, err = %v", status, err)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Fatalf("missing credential injected %q", got)
	}
}

func TestStripInboundRemovesCredentials(t *testing.T) {
	req := httptest.NewRequest("GET", "http://127.0.0.1", nil)
	req.Header.Set("Authorization", "Bearer client")
	req.Header.Set("Proxy-Authorization", "Basic client")
	req.Header.Set("Cookie", "session=client")
	req.Header.Set("X-Engine-Key", "client")
	StripInbound(req.Header, Config{Scheme: SchemeHeader, Credential: "engine.test.api_key", Header: "X-Engine-Key"})
	for _, name := range []string{"Authorization", "Proxy-Authorization", "Cookie", "X-Engine-Key"} {
		if got := req.Header.Get(name); got != "" {
			t.Fatalf("%s leaked as %q", name, got)
		}
	}
}
