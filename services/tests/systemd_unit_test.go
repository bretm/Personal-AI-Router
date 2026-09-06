// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"os"
	"strings"
	"testing"
)

func TestSystemdUnitLoadsEncryptedLemonadeCredential(t *testing.T) {
	unit, err := os.ReadFile("../installer/linux/nvpair.service.tmpl")
	if err != nil {
		t.Fatalf("read systemd unit template: %v", err)
	}
	text := string(unit)
	for _, required := range []string{
		"LoadCredentialEncrypted=lemonade-api-key",
		"$${CREDENTIALS_DIRECTORY}/lemonade-api-key",
		"LEMONADE_API_KEY=",
		"nvpair-ui-broker\" --headless",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("systemd unit template is missing %q", required)
		}
	}
	for _, forbidden := range []string{
		"EnvironmentFile=",
		"LEMONADE_ADMIN_API_KEY",
	} {
		if strings.Contains(text, forbidden) {
			t.Errorf("systemd unit template must not contain %q", forbidden)
		}
	}
}
