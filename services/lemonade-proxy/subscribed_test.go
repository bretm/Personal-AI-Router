// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/noderec"
)

// TestSubscribedToNodeUsesLemonadeService locks the relay projection to the
// Lemonade service and model inventory. A Lemonade proxy subscribed to the LM
// Studio service sees neither its local engine nor Lemonade-capable peers, so
// model-bearing requests are rejected before forwarding.
func TestSubscribedToNodeUsesLemonadeService(t *testing.T) {
	lemonade := noderec.DirectoryNode{
		HostUUID: "uuid-lemonade",
		Name:     "lemonade-node",
		IP:       "10.0.0.5",
		Models:   []string{"lemonade-model"},
		ModelsByEngine: map[string][]string{
			"lemonade": {"lemonade-model"},
		},
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceLemonade: {Port: 13306},
		},
	}

	got, ok := subscribedToNode(lemonade)
	if !ok {
		t.Fatal("node with Lemonade service and IP should project")
	}
	if got.ID != "uuid-lemonade" || got.Port != 13306 || got.IP != "10.0.0.5" {
		t.Fatalf("unexpected Lemonade projection: %+v", got)
	}
	if len(got.Models) != 1 || got.Models[0] != "lemonade-model" {
		t.Fatalf("projection Models = %v, want [lemonade-model]", got.Models)
	}

	lmStudioOnly := noderec.DirectoryNode{
		HostUUID: "uuid-lmstudio",
		Name:     "lmstudio-node",
		IP:       "10.0.0.6",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceLMStudio: {Port: 1234},
		},
	}
	if _, ok := subscribedToNode(lmStudioOnly); ok {
		t.Fatal("node without Lemonade service should not project")
	}
}
