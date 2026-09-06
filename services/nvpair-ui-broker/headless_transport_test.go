// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"testing"
	"time"
)

func TestHeadlessTransportKeepsBrokerAliveUntilClosed(t *testing.T) {
	transport := newHeadlessTransport()
	readDone := make(chan error, 1)
	go func() {
		var buf [1]byte
		_, err := transport.Read(buf[:])
		readDone <- err
	}()

	select {
	case err := <-readDone:
		t.Fatalf("Read returned before Close: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	if err := transport.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case err := <-readDone:
		if err != io.EOF {
			t.Fatalf("Read after Close error = %v, want EOF", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Read remained blocked after Close")
	}

	if err := transport.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestHeadlessTransportDiscardsControllerNotifications(t *testing.T) {
	transport := newHeadlessTransport()
	t.Cleanup(func() { _ = transport.Close() })

	payload := []byte("controller notification")
	n, err := transport.Write(payload)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if n != len(payload) {
		t.Fatalf("Write count = %d, want %d", n, len(payload))
	}
}
