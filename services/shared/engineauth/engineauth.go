// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package engineauth resolves and applies node-local engine credentials.
package engineauth

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"

	"github.com/99designs/keyring"
)

// ServiceName is a stable native-store identifier, not display copy. In
// Secret Service, labels containing spaces can be normalized to a different
// D-Bus collection path; clients then fail to reopen the collection and create
// suffixed duplicates on later writes.
const ServiceName = "nvpair"

type Scheme string

const (
	SchemeNone   Scheme = "none"
	SchemeBearer Scheme = "bearer-token"
	SchemeHeader Scheme = "header"
)

// Config is public manifest metadata. It deliberately carries a reference, not
// a secret, so it is safe to pass between PAIR's local processes.
type Config struct {
	Scheme      Scheme `json:"scheme,omitempty"`
	Credential  string `json:"credential,omitempty"`
	Environment string `json:"environment,omitempty"`
	Header      string `json:"header,omitempty"`
}

type Source string

const (
	SourceMissing Source = "missing"
	SourceEnv     Source = "environment"
	SourceStore   Source = "secure_store"
)

type Status struct {
	Configured bool   `json:"configured"`
	Source     Source `json:"source"`
}

var (
	credentialRE  = regexp.MustCompile(`^engine\.[A-Za-z0-9._-]+\.api_key$`)
	environmentRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	headerRE      = regexp.MustCompile(`^[!#$%&'*+.^_` + "`" + `|~0-9A-Za-z-]+$`)
)

func (c Config) Normalized() Config {
	if c.Scheme == "" {
		c.Scheme = SchemeNone
	}
	return c
}

func (c Config) Validate() error {
	c = c.Normalized()
	switch c.Scheme {
	case SchemeNone:
		if c.Credential != "" || c.Environment != "" || c.Header != "" {
			return errors.New("unauthenticated engine cannot declare credential fields")
		}
		return nil
	case SchemeBearer, SchemeHeader:
		if !credentialRE.MatchString(c.Credential) {
			return fmt.Errorf("invalid credential reference %q", c.Credential)
		}
		if c.Environment != "" && !environmentRE.MatchString(c.Environment) {
			return fmt.Errorf("invalid credential environment %q", c.Environment)
		}
		if c.Scheme == SchemeHeader && !headerRE.MatchString(c.Header) {
			return fmt.Errorf("invalid credential header %q", c.Header)
		}
		if c.Scheme == SchemeBearer && c.Header != "" {
			return errors.New("bearer-token auth cannot declare a custom header")
		}
		return nil
	default:
		return fmt.Errorf("unsupported auth scheme %q", c.Scheme)
	}
}

type Store interface {
	Get(string) (string, error)
	Set(string, string) error
	Remove(string) error
}

type nativeStore struct{ ring keyring.Keyring }

type storeOpener func() (Store, error)

// deferredStore keeps process startup and environment-only authentication away
// from the OS credential service. The backend is opened at most once, on the
// first operation that actually needs native storage.
type deferredStore struct {
	once   sync.Once
	opener storeOpener
	store  Store
	err    error
}

func newDeferredStore(opener storeOpener) Store {
	return &deferredStore{opener: opener}
}

func (s *deferredStore) load() (Store, error) {
	s.once.Do(func() {
		s.store, s.err = s.opener()
	})
	return s.store, s.err
}

func (s *deferredStore) Get(ref string) (string, error) {
	store, err := s.load()
	if err != nil {
		return "", err
	}
	return store.Get(ref)
}

func (s *deferredStore) Set(ref, value string) error {
	store, err := s.load()
	if err != nil {
		return err
	}
	return store.Set(ref, value)
}

func (s *deferredStore) Remove(ref string) error {
	store, err := s.load()
	if err != nil {
		return err
	}
	return store.Remove(ref)
}

// NativeStore returns a deferred handle. Constructing it never contacts,
// unlocks, or creates an OS credential store; only Get, Set, or Remove does.
func NativeStore() Store {
	return newDeferredStore(openNativeStore)
}

func openNativeStore() (Store, error) {
	allowed := []keyring.BackendType{keyring.SecretServiceBackend, keyring.KWalletBackend}
	switch runtime.GOOS {
	case "darwin":
		allowed = []keyring.BackendType{keyring.KeychainBackend}
	case "windows":
		allowed = []keyring.BackendType{keyring.WinCredBackend}
	}
	ring, err := keyring.Open(keyring.Config{ServiceName: ServiceName, AllowedBackends: allowed})
	if err != nil {
		return nil, fmt.Errorf("open native credential store: %w", err)
	}
	return nativeStore{ring: ring}, nil
}

func (s nativeStore) Get(ref string) (string, error) {
	item, err := s.ring.Get(ref)
	if err != nil {
		return "", err
	}
	return string(item.Data), nil
}

func (s nativeStore) Set(ref, value string) error {
	return s.ring.Set(keyring.Item{Key: ref, Data: []byte(value), Label: ref})
}

func (s nativeStore) Remove(ref string) error { return s.ring.Remove(ref) }

// IsNotFound reports whether a credential-store operation failed because the
// requested secret does not exist. Callers use it to make status and clear
// operations idempotent without depending directly on the keyring package.
func IsNotFound(err error) bool { return errors.Is(err, keyring.ErrKeyNotFound) }

func Resolve(c Config, store Store) (string, Status, error) {
	c = c.Normalized()
	if err := c.Validate(); err != nil {
		return "", Status{}, err
	}
	if c.Scheme == SchemeNone {
		return "", Status{Source: SourceMissing}, nil
	}
	if c.Environment != "" {
		if value := os.Getenv(c.Environment); value != "" {
			return value, Status{Configured: true, Source: SourceEnv}, nil
		}
	}
	if store == nil {
		return "", Status{Source: SourceMissing}, nil
	}
	value, err := store.Get(c.Credential)
	if err != nil {
		if IsNotFound(err) {
			return "", Status{Source: SourceMissing}, nil
		}
		return "", Status{Source: SourceMissing}, fmt.Errorf("read credential status: %w", err)
	}
	if value == "" {
		return "", Status{Source: SourceMissing}, nil
	}
	return value, Status{Configured: true, Source: SourceStore}, nil
}

// Apply resolves the configured local secret and adds it to req. Callers must
// use it only after deciding that req is a terminal local-engine request.
func Apply(req *http.Request, c Config, store Store) (Status, error) {
	value, status, err := Resolve(c, store)
	if err != nil || c.Normalized().Scheme == SchemeNone {
		return status, err
	}
	if !status.Configured {
		return status, errors.New("engine credential is required but not configured")
	}
	switch c.Normalized().Scheme {
	case SchemeBearer:
		req.Header.Set("Authorization", "Bearer "+value)
	case SchemeHeader:
		req.Header.Set(c.Header, value)
	}
	return status, nil
}

// StripInbound removes credentials supplied by a client. PAIR owns its
// downstream credential and must not relay caller credentials to another node.
func StripInbound(h http.Header, c Config) {
	h.Del("Authorization")
	h.Del("Proxy-Authorization")
	h.Del("Cookie")
	if c.Normalized().Scheme == SchemeHeader {
		h.Del(c.Header)
	}
}

func IsCredentialRef(ref string) bool { return credentialRE.MatchString(strings.TrimSpace(ref)) }
