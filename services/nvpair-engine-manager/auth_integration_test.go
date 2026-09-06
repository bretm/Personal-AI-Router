// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/engineauth"
)

type testCredentialStore map[string]string

func (s testCredentialStore) Get(key string) (string, error) {
	value, ok := s[key]
	if !ok {
		return "", errors.New("not found")
	}
	return value, nil
}
func (s testCredentialStore) Set(key, value string) error { s[key] = value; return nil }
func (s testCredentialStore) Remove(key string) error     { delete(s, key); return nil }

func TestAuthenticatedEngineControlPlane(t *testing.T) {
	const secret = "engine-test-secret"
	var mu sync.Mutex
	seen := map[string]int{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+secret {
			http.Error(w, "credential required", http.StatusUnauthorized)
			return
		}
		mu.Lock()
		seen[r.URL.Path]++
		mu.Unlock()
		switch r.URL.Path {
		case "/pull":
			_, _ = w.Write([]byte("{\"status\":\"complete\"}\n"))
		default:
			_, _ = w.Write([]byte("{\"models\":[]}"))
		}
	}))
	defer server.Close()

	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}
	manifest := validManifest()
	manifest.Engine = "example"
	manifest.Auth = engineauth.Config{
		Scheme:     engineauth.SchemeBearer,
		Credential: "engine.example.api_key",
	}
	platform := manifest.Platforms["linux/amd64"]
	delete(manifest.Platforms, "linux/amd64")
	platform.Runtime.Port = port
	platform.Runtime.Ready = &Probe{HTTP: server.URL + "/health", Status: http.StatusOK}
	manifest.Platforms[authHostKey()] = platform
	manifest.Actions = map[string]Action{
		"list_models":  {HTTP: &ActionHTTP{Method: http.MethodGet, Path: "/models"}},
		"delete_model": {HTTP: &ActionHTTP{Method: http.MethodDelete, Path: "/manage"}},
		"pull_model":   {HTTP: &ActionHTTP{Method: http.MethodPost, Path: "/pull"}},
	}

	executor := newTestExecutor(t, &manifest)
	executor.credentialStore = testCredentialStore{"engine.example.api_key": secret}
	state, err := executor.state("example")
	if err != nil {
		t.Fatal(err)
	}
	state.running = true
	state.port = port
	if !executor.probe(context.Background(), platform.Runtime.Ready, port, &manifest) {
		t.Fatal("authenticated health probe failed")
	}
	for _, action := range []string{"list_models", "delete_model"} {
		if _, err := executor.Action(context.Background(), "example", action, nil); err != nil {
			t.Fatalf("%s: %v", action, err)
		}
	}
	if _, err := executor.PullModelStream(context.Background(), "example", "model-a", nil); err != nil {
		t.Fatalf("pull_model: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{"/health", "/models", "/manage", "/pull"} {
		if seen[path] == 0 {
			t.Errorf("authenticated request did not reach %s", path)
		}
	}
}

func TestMissingEngineCredentialIsActionableAndSecretFree(t *testing.T) {
	manifest := validManifest()
	manifest.Engine = "example"
	manifest.Auth = engineauth.Config{Scheme: engineauth.SchemeBearer, Credential: "engine.example.api_key"}
	manifest.Platforms[authHostKey()] = manifest.Platforms["linux/amd64"]
	executor := newTestExecutor(t, &manifest)
	executor.credentialStore = nil
	state, err := executor.state("example")
	if err != nil {
		t.Fatal(err)
	}
	state.running = true
	_, err = executor.Action(context.Background(), "example", "list_models", nil)
	if err == nil || !strings.Contains(err.Error(), "credential is required but not configured") {
		t.Fatalf("missing credential error = %v", err)
	}
}

func TestLemonadeCredentialCoversHealthInventoryAndManagement(t *testing.T) {
	const secret = "lemonade-test-secret"
	t.Setenv("LEMONADE_API_KEY", secret)
	var mu sync.Mutex
	seen := map[string]int{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+secret {
			http.Error(w, "credential required", http.StatusUnauthorized)
			return
		}
		mu.Lock()
		seen[r.URL.Path]++
		mu.Unlock()
		switch r.URL.Path {
		case "/v1/health":
			_, _ = w.Write([]byte(`{"all_models_loaded":[{"model_name":"model-a"}]}`))
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"model-a"}]}`))
		case "/v1/pull":
			_, _ = w.Write([]byte("{\"status\":\"complete\"}\n"))
		default:
			_, _ = w.Write([]byte(`{"ok":true}`))
		}
	}))
	defer server.Close()
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}

	reg := NewRegistry()
	if err := reg.LoadFS(bundledManifests, "manifests"); err != nil {
		t.Fatal(err)
	}
	bundled, ok := reg.Get("lemonade")
	if !ok {
		t.Fatal("lemonade manifest not loaded")
	}
	manifest := *bundled
	platform := bundled.Platforms[authHostKey()]
	platform.Runtime.Port = port
	manifest.Platforms = map[string]Platform{authHostKey(): platform}

	executor := newTestExecutor(t, &manifest)
	// The environment value must win over a conflicting native-store value.
	executor.credentialStore = testCredentialStore{"engine.lemonade.api_key": "wrong-store-secret"}
	state, err := executor.state("lemonade")
	if err != nil {
		t.Fatal(err)
	}
	state.running = true
	state.port = port
	if !executor.probe(context.Background(), platform.Runtime.Ready, port, &manifest) {
		t.Fatal("authenticated readiness probe failed")
	}
	if !executor.probe(context.Background(), platform.Runtime.Health, port, &manifest) {
		t.Fatal("authenticated health probe failed")
	}
	if _, err := executor.Action(context.Background(), "lemonade", "list_models", nil); err != nil {
		t.Fatalf("list_models: %v", err)
	}
	if _, err := executor.Action(context.Background(), "lemonade", "loaded_models", nil); err != nil {
		t.Fatalf("loaded_models: %v", err)
	}
	if _, err := executor.ModelLoad(context.Background(), "lemonade", "model-a"); err != nil {
		t.Fatalf("load_model: %v", err)
	}
	if _, err := executor.ModelUnload(context.Background(), "lemonade", "model-a"); err != nil {
		t.Fatalf("unload_model: %v", err)
	}
	if _, err := executor.ModelDelete(context.Background(), "lemonade", "model-a"); err != nil {
		t.Fatalf("delete_model: %v", err)
	}
	if _, err := executor.PullModelStream(context.Background(), "lemonade", "model-a", nil); err != nil {
		t.Fatalf("pull_model: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{"/v1/health", "/v1/models", "/v1/load", "/v1/unload", "/v1/delete", "/v1/pull"} {
		if seen[path] == 0 {
			t.Errorf("authenticated request did not reach %s", path)
		}
	}
}

func authHostKey() string {
	return runtime.GOOS + "/" + runtime.GOARCH
}
