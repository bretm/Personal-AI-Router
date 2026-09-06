// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package meshauth owns PAIR's signed, cluster-wide client-token verifier.
package meshauth

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"nvpair-shared/engineauth"
)

const fileName = "mesh-auth.json"

type Client struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Hash    string `json:"hash"`
	Revoked bool   `json:"revoked,omitempty"`
}

type Document struct {
	ClusterID string   `json:"clusterId"`
	Revision  uint64   `json:"revision"`
	Owner     string   `json:"owner"`
	Clients   []Client `json:"clients"`
	Signature string   `json:"signature"`
}

type Registry struct {
	dir string
	mu  sync.RWMutex
}

func New(dir string) *Registry   { return &Registry{dir: dir} }
func (r *Registry) Path() string { return filepath.Join(r.dir, fileName) }

// Required reports whether this node has entered mesh-auth mode. A corrupt
// registry remains required and therefore fails closed during validation.
func (r *Registry) Required() bool {
	_, err := os.Stat(r.Path())
	return err == nil || !errors.Is(err, os.ErrNotExist)
}

func ownerRef(clusterID string) string { return "pair.mesh." + clusterID + ".owner_key" }

type Status struct {
	Initialized   bool   `json:"initialized"`
	Administrator bool   `json:"administrator"`
	ClusterID     string `json:"clusterId,omitempty"`
	Revision      uint64 `json:"revision,omitempty"`
}

func canonical(d Document) ([]byte, error) {
	d.Signature = ""
	sort.Slice(d.Clients, func(i, j int) bool { return d.Clients[i].ID < d.Clients[j].ID })
	return json.Marshal(d)
}

func sign(d *Document, private ed25519.PrivateKey) error {
	payload, err := canonical(*d)
	if err != nil {
		return err
	}
	d.Signature = base64.RawStdEncoding.EncodeToString(ed25519.Sign(private, payload))
	return nil
}

func Verify(d Document) error {
	pub, err := base64.RawStdEncoding.DecodeString(d.Owner)
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return errors.New("invalid mesh auth owner key")
	}
	sig, err := base64.RawStdEncoding.DecodeString(d.Signature)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return errors.New("invalid mesh auth signature")
	}
	payload, err := canonical(d)
	if err != nil {
		return err
	}
	if !ed25519.Verify(ed25519.PublicKey(pub), payload, sig) {
		return errors.New("mesh auth signature verification failed")
	}
	return nil
}

func (r *Registry) Load() (Document, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b, err := os.ReadFile(r.Path())
	if err != nil {
		return Document{}, err
	}
	var d Document
	if err := json.Unmarshal(b, &d); err != nil {
		return Document{}, err
	}
	if err := Verify(d); err != nil {
		return Document{}, err
	}
	return d, nil
}

func writeAtomic(path string, d Document) error {
	b, err := json.Marshal(d)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (r *Registry) Bootstrap(clusterID string, store engineauth.Store) (Document, error) {
	if clusterID == "" {
		return Document{}, errors.New("cluster identity is required")
	}
	if store == nil {
		return Document{}, errors.New("native credential store is unavailable")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if d, err := r.loadLocked(); err == nil {
		if d.ClusterID != clusterID {
			return Document{}, errors.New("mesh auth belongs to another cluster")
		}
		if _, err := r.ownerKey(d, store); err != nil {
			return Document{}, err
		}
		return d, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return Document{}, fmt.Errorf("read existing mesh auth registry: %w", err)
	}
	pub, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return Document{}, err
	}
	if err := store.Set(ownerRef(clusterID), base64.RawStdEncoding.EncodeToString(private)); err != nil {
		return Document{}, err
	}
	rollbackOwner := func(cause error) error {
		if removeErr := store.Remove(ownerRef(clusterID)); removeErr != nil && !engineauth.IsNotFound(removeErr) {
			return errors.Join(cause, fmt.Errorf("remove uncommitted mesh administrator key: %w", removeErr))
		}
		return cause
	}
	d := Document{ClusterID: clusterID, Revision: 1, Owner: base64.RawStdEncoding.EncodeToString(pub), Clients: []Client{}}
	if err := sign(&d, private); err != nil {
		return Document{}, rollbackOwner(err)
	}
	if err := writeAtomic(r.Path(), d); err != nil {
		return Document{}, rollbackOwner(err)
	}
	return d, nil
}

// Status returns public registry metadata plus whether this node holds the
// corresponding private owner key. It never returns private key material or a
// client token/hash.
func (r *Registry) Status(store engineauth.Store) (Status, error) {
	d, err := r.Load()
	if errors.Is(err, os.ErrNotExist) {
		return Status{}, nil
	}
	if err != nil {
		return Status{}, err
	}
	_, ownerErr := r.ownerKey(d, store)
	return Status{
		Initialized:   true,
		Administrator: ownerErr == nil,
		ClusterID:     d.ClusterID,
		Revision:      d.Revision,
	}, nil
}

// Clear removes this cluster's registry and any locally held owner key. A
// non-administrator normally has no owner key, so a missing secret is not an
// error. The registry is removed last, preserving enough information for a
// failed durable teardown to retry.
func (r *Registry) Clear(clusterID string, store engineauth.Store) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, err := r.loadLocked()
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read mesh auth registry: %w", err)
	}
	if clusterID != "" && clusterID != d.ClusterID {
		return errors.New("mesh auth belongs to another cluster")
	}
	if store != nil {
		if err := store.Remove(ownerRef(d.ClusterID)); err != nil && !engineauth.IsNotFound(err) {
			return fmt.Errorf("remove mesh administrator key: %w", err)
		}
	}
	if err := os.Remove(r.Path()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove mesh auth registry: %w", err)
	}
	return nil
}

// Discard removes only the public signed registry. It is used to roll back a
// joiner's provisional pairing state; it never contacts the credential store.
func (r *Registry) Discard(clusterID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, err := r.loadLocked()
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read mesh auth registry: %w", err)
	}
	if clusterID != "" && clusterID != d.ClusterID {
		return errors.New("mesh auth belongs to another cluster")
	}
	if err := os.Remove(r.Path()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove mesh auth registry: %w", err)
	}
	return nil
}

func (r *Registry) ownerKey(d Document, store engineauth.Store) (ed25519.PrivateKey, error) {
	if store == nil {
		return nil, errors.New("native credential store is unavailable")
	}
	raw, err := store.Get(ownerRef(d.ClusterID))
	if err != nil {
		return nil, errors.New("this node is not the mesh administrator")
	}
	b, err := base64.RawStdEncoding.DecodeString(raw)
	if err != nil || len(b) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid local mesh administrator key")
	}
	return ed25519.PrivateKey(b), nil
}

func validLabel(label string) bool {
	return label != "" && len(label) <= 80 && !strings.ContainsAny(label, "\r\n")
}

func (r *Registry) AddClient(label, token string, store engineauth.Store) (Client, error) {
	if !validLabel(label) || token == "" {
		return Client{}, errors.New("client label and token are required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	d, err := r.loadLocked()
	if err != nil {
		return Client{}, err
	}
	private, err := r.ownerKey(d, store)
	if err != nil {
		return Client{}, err
	}
	for _, c := range d.Clients {
		if c.Label == label && !c.Revoked {
			return Client{}, errors.New("an active client already has that label")
		}
	}
	idBytes := make([]byte, 8)
	if _, err := rand.Read(idBytes); err != nil {
		return Client{}, err
	}
	h := sha256.Sum256([]byte(token))
	c := Client{ID: hex.EncodeToString(idBytes), Label: label, Hash: hex.EncodeToString(h[:])}
	d.Clients = append(d.Clients, c)
	d.Revision++
	if err := sign(&d, private); err != nil {
		return Client{}, err
	}
	if err := writeAtomic(r.Path(), d); err != nil {
		return Client{}, err
	}
	return c, nil
}

// CreateClient mints a high-entropy client token and persists only its hash.
// The token is returned to the local administrator exactly once so it can be
// delivered to the client out of band; it is not recoverable from the registry.
func (r *Registry) CreateClient(label string, store engineauth.Store) (Client, string, error) {
	token, err := ClientToken()
	if err != nil {
		return Client{}, "", err
	}
	client, err := r.AddClient(label, token, store)
	if err != nil {
		return Client{}, "", err
	}
	return client, token, nil
}

func (r *Registry) RevokeClient(id string, store engineauth.Store) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	d, err := r.loadLocked()
	if err != nil {
		return err
	}
	private, err := r.ownerKey(d, store)
	if err != nil {
		return err
	}
	found := false
	for i := range d.Clients {
		if d.Clients[i].ID == id {
			d.Clients[i].Revoked = true
			found = true
		}
	}
	if !found {
		return errors.New("mesh client was not found")
	}
	d.Revision++
	if err := sign(&d, private); err != nil {
		return err
	}
	return writeAtomic(r.Path(), d)
}

func (r *Registry) loadLocked() (Document, error) {
	b, err := os.ReadFile(r.Path())
	if err != nil {
		return Document{}, err
	}
	var d Document
	if err := json.Unmarshal(b, &d); err != nil {
		return Document{}, err
	}
	if err := Verify(d); err != nil {
		return Document{}, err
	}
	return d, nil
}

func (r *Registry) ValidateBearer(token string) (Client, bool) {
	d, err := r.Load()
	if err != nil {
		return Client{}, false
	}
	h := sha256.Sum256([]byte(token))
	want := hex.EncodeToString(h[:])
	for _, c := range d.Clients {
		if c.Revoked {
			continue
		}
		if subtle.ConstantTimeCompare([]byte(c.Hash), []byte(want)) == 1 {
			return c, true
		}
	}
	return Client{}, false
}

func (r *Registry) Import(d Document) (bool, error) {
	if err := Verify(d); err != nil {
		return false, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	current, err := r.loadLocked()
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return false, fmt.Errorf("read current mesh auth registry: %w", err)
	}
	if err == nil && current.ClusterID != d.ClusterID {
		return false, errors.New("mesh auth cluster does not match")
	}
	if err == nil && current.Owner != d.Owner {
		return false, errors.New("mesh auth owner does not match")
	}
	if err == nil && current.Revision >= d.Revision {
		return false, nil
	}
	if err := writeAtomic(r.Path(), d); err != nil {
		return false, err
	}
	return true, nil
}

func (r *Registry) Export() (Document, error) { return r.Load() }

func ParseBearer(header string) (string, bool) {
	parts := strings.Fields(header)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		return "", false
	}
	return parts[1], true
}

func ClientToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generate client token: %w", err)
	}
	return "pair_" + base64.RawURLEncoding.EncodeToString(b), nil
}
