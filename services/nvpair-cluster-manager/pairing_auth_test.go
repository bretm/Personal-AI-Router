// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"

	"nvpair-shared/meshauth"
)

type pairingCredentialStore map[string]string

func (s pairingCredentialStore) Get(key string) (string, error) {
	value, ok := s[key]
	if !ok {
		return "", errPairingCommitStale
	}
	return value, nil
}

func (s pairingCredentialStore) Set(key, value string) error {
	s[key] = value
	return nil
}

func (s pairingCredentialStore) Remove(key string) error {
	delete(s, key)
	return nil
}

func TestPairingInfoCarriesAuthenticatedMeshRegistry(t *testing.T) {
	m := newTestManagerPort(t, 15142)
	m.credentialStore = pairingCredentialStore{}
	if _, err := m.meshAuth.Bootstrap("cluster-1", m.credentialStore); err != nil {
		t.Fatal(err)
	}

	raw, err := json.Marshal(m.localPairingInfo("127.0.0.1:15142"))
	if err != nil {
		t.Fatal(err)
	}
	info, _, err := parsePairingInfo(raw)
	if err != nil {
		t.Fatal(err)
	}
	if info.MeshAuth == nil || info.MeshAuth.ClusterID != "cluster-1" {
		t.Fatalf("pairing info mesh auth = %+v", info.MeshAuth)
	}
	if err := meshauth.Verify(*info.MeshAuth); err != nil {
		t.Fatal(err)
	}
}

func TestJoinerImportsMeshRegistryBeforeAdmission(t *testing.T) {
	owner := newTestManagerPort(t, 15144)
	owner.credentialStore = pairingCredentialStore{}
	if _, err := owner.meshAuth.Bootstrap("cluster-1", owner.credentialStore); err != nil {
		t.Fatal(err)
	}
	_, token, err := owner.meshAuth.CreateClient("hermes", owner.credentialStore)
	if err != nil {
		t.Fatal(err)
	}

	joiner := testManagerAt(t, t.TempDir(), 15145)
	imported, err := joiner.importPairingMeshAuth(owner.localPairingInfo("127.0.0.1:15144"), "cluster-1")
	if err != nil || !imported {
		t.Fatalf("imported = %v, err = %v", imported, err)
	}
	if _, ok := joiner.meshAuth.ValidateBearer(token); !ok {
		t.Fatal("joiner did not validate client from pairing registry")
	}
}

func TestPairingInfoRejectsMeshRegistryForAnotherCluster(t *testing.T) {
	m := newTestManagerPort(t, 15143)
	foreignStore := pairingCredentialStore{}
	foreign, err := meshauth.New(t.TempDir()).Bootstrap("cluster-2", foreignStore)
	if err != nil {
		t.Fatal(err)
	}
	info := m.localPairingInfo("127.0.0.1:15143")
	info.MeshAuth = &foreign
	raw, err := json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := parsePairingInfo(raw); err == nil {
		t.Fatal("pairing info accepted a mesh registry from another cluster")
	}
}
