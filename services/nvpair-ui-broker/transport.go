// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"os"
	"sync"
)

type stdioTransport struct {
	io.Reader
	io.Writer
}

func newStdioTransport() io.ReadWriteCloser {
	return &stdioTransport{
		Reader: os.Stdin,
		Writer: os.Stdout,
	}
}

func (s *stdioTransport) Close() error {
	return nil
}

// headlessTransport keeps the broker's controller side open without exposing a
// control channel. Reads block until Close and writes are discarded, allowing
// the broker to reuse its normal supervision lifecycle while systemd owns the
// process. Operational output remains on stderr and is captured by the journal.
type headlessTransport struct {
	done chan struct{}
	once sync.Once
}

func newHeadlessTransport() io.ReadWriteCloser {
	return &headlessTransport{done: make(chan struct{})}
}

func (t *headlessTransport) Read([]byte) (int, error) {
	<-t.done
	return 0, io.EOF
}

func (t *headlessTransport) Write(p []byte) (int, error) {
	return len(p), nil
}

func (t *headlessTransport) Close() error {
	t.once.Do(func() { close(t.done) })
	return nil
}
