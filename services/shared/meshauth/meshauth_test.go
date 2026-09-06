// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package meshauth

import (
	"errors"
	"path/filepath"
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

type countingStore struct {
	gets    int
	sets    int
	removes int
}

func (s *countingStore) Get(string) (string, error) {
	s.gets++
	return "", errors.New("not found")
}

func (s *countingStore) Set(string, string) error {
	s.sets++
	return nil
}

func (s *countingStore) Remove(string) error {
	s.removes++
	return nil
}

func TestClearWithoutRegistryDoesNotAccessCredentialStore(t *testing.T) {
	r := New(t.TempDir())
	store := &countingStore{}
	if err := r.Clear("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	if store.gets != 0 || store.sets != 0 || store.removes != 0 {
		t.Fatalf("clear without registry accessed credential store: gets=%d sets=%d removes=%d", store.gets, store.sets, store.removes)
	}
}

func TestDiscardNeverAccessesCredentialStore(t *testing.T) {
	r := New(t.TempDir())
	store := memoryStore{}
	if _, err := r.Bootstrap("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	if err := r.Discard("cluster-a"); err != nil {
		t.Fatal(err)
	}
	if r.Required() {
		t.Fatal("discard retained public registry")
	}
	if len(store) != 1 {
		t.Fatal("discard changed native credential entries")
	}
}

func TestBootstrapRemovesOwnerKeyWhenRegistryCommitFails(t *testing.T) {
	store := memoryStore{}
	r := New(filepath.Join(t.TempDir(), "missing"))
	if _, err := r.Bootstrap("cluster-a", store); err == nil {
		t.Fatal("bootstrap unexpectedly committed without a registry directory")
	}
	if len(store) != 0 {
		t.Fatal("failed bootstrap left an orphaned administrator key")
	}
}

func TestRegistryClientLifecycle(t *testing.T) {
	r := New(t.TempDir())
	store := memoryStore{}
	if _, err := r.Bootstrap("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	client, err := r.AddClient("hermes", "pair_test_token", store)
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := r.ValidateBearer("pair_test_token"); !ok || got.ID != client.ID {
		t.Fatalf("validated client = %+v, %v", got, ok)
	}
	if err := r.RevokeClient(client.ID, store); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.ValidateBearer("pair_test_token"); ok {
		t.Fatal("revoked token remained valid")
	}
}

func TestCreateClientReturnsTokenOnlyAtCreation(t *testing.T) {
	r := New(t.TempDir())
	store := memoryStore{}
	if _, err := r.Bootstrap("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	client, token, err := r.CreateClient("hermes", store)
	if err != nil {
		t.Fatal(err)
	}
	if token == "" || token[:5] != "pair_" {
		t.Fatalf("token = %q", token)
	}
	if got, ok := r.ValidateBearer(token); !ok || got.ID != client.ID {
		t.Fatalf("generated token validated as %+v, %v", got, ok)
	}
	doc, err := r.Export()
	if err != nil {
		t.Fatal(err)
	}
	if len(doc.Clients) != 1 || doc.Clients[0].Hash == token {
		t.Fatalf("registry retained raw token: %+v", doc.Clients)
	}
}

func TestImportRejectsTamperedDocument(t *testing.T) {
	store := memoryStore{}
	from := New(t.TempDir())
	doc, err := from.Bootstrap("cluster-a", store)
	if err != nil {
		t.Fatal(err)
	}
	doc.Revision++
	if _, err := New(t.TempDir()).Import(doc); err == nil {
		t.Fatal("tampered document was accepted")
	}
}

func TestStatusAndSignedRegistryConvergence(t *testing.T) {
	store := memoryStore{}
	owner := New(t.TempDir())
	doc, err := owner.Bootstrap("cluster-a", store)
	if err != nil {
		t.Fatal(err)
	}
	status, err := owner.Status(store)
	if err != nil || !status.Initialized || !status.Administrator || status.Revision != 1 {
		t.Fatalf("owner status = %+v, err = %v", status, err)
	}

	peer := New(t.TempDir())
	if changed, err := peer.Import(doc); err != nil || !changed {
		t.Fatalf("initial import changed = %v, err = %v", changed, err)
	}
	peerStatus, err := peer.Status(memoryStore{})
	if err != nil || !peerStatus.Initialized || peerStatus.Administrator {
		t.Fatalf("peer status = %+v, err = %v", peerStatus, err)
	}

	_, token, err := owner.CreateClient("hermes", store)
	if err != nil {
		t.Fatal(err)
	}
	updated, err := owner.Export()
	if err != nil {
		t.Fatal(err)
	}
	if changed, err := peer.Import(updated); err != nil || !changed {
		t.Fatalf("updated import changed = %v, err = %v", changed, err)
	}
	if _, ok := peer.ValidateBearer(token); !ok {
		t.Fatal("peer did not converge on the signed client registry")
	}
}

func TestImportRejectsOwnerReplacement(t *testing.T) {
	legitimate, err := New(t.TempDir()).Bootstrap("cluster-a", memoryStore{})
	if err != nil {
		t.Fatal(err)
	}
	peer := New(t.TempDir())
	if _, err := peer.Import(legitimate); err != nil {
		t.Fatal(err)
	}

	attackerRegistry := New(t.TempDir())
	attackerStore := memoryStore{}
	if _, err := attackerRegistry.Bootstrap("cluster-a", attackerStore); err != nil {
		t.Fatal(err)
	}
	if _, _, err := attackerRegistry.CreateClient("forged", attackerStore); err != nil {
		t.Fatal(err)
	}
	forged, err := attackerRegistry.Export()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := peer.Import(forged); err == nil {
		t.Fatal("registry signed by a replacement owner was accepted")
	}
}

func TestClearRemovesRegistryAndOwnerKey(t *testing.T) {
	store := memoryStore{}
	r := New(t.TempDir())
	if _, err := r.Bootstrap("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	if err := r.Clear("cluster-a", store); err != nil {
		t.Fatal(err)
	}
	if r.Required() {
		t.Fatal("registry remained required after clear")
	}
	if _, ok := store[ownerRef("cluster-a")]; ok {
		t.Fatal("owner key remained after clear")
	}
}
