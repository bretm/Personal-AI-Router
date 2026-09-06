// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"testing"
)

type countingCredentialStore struct {
	gets    int
	sets    int
	removes int
}

func (s *countingCredentialStore) Get(string) (string, error) {
	s.gets++
	return "", errors.New("not found")
}

func (s *countingCredentialStore) Set(string, string) error {
	s.sets++
	return nil
}

func (s *countingCredentialStore) Remove(string) error {
	s.removes++
	return nil
}

func TestFoundClusterDoesNotCreateCredentialStoreEntries(t *testing.T) {
	m := testManagerAt(t, t.TempDir(), 15141)
	store := &countingCredentialStore{}
	m.credentialStore = store

	if _, _, err := m.foundCluster("Lab"); err != nil {
		t.Fatal(err)
	}
	if store.gets != 0 || store.sets != 0 || store.removes != 0 {
		t.Fatalf("ordinary cluster creation accessed credential store: gets=%d sets=%d removes=%d", store.gets, store.sets, store.removes)
	}
}
