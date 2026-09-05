// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"strings"

	"nvpair-shared/applog"
	"nvpair-shared/noderec"
)

// lemonade-proxy uses the same broker-to-proxy control protocol as the other
// inference proxies. It remains a separate process so its discovery identity,
// workload attribution, and operational errors stay unambiguous.
func (b *Broker) setLemonadeProxy(p *proxyProcess) {
	b.workersMu.Lock()
	b.lemonadeProxy = p
	b.workersMu.Unlock()
}

func (b *Broker) getLemonadeProxy() *proxyProcess {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.lemonadeProxy
}

func (b *Broker) configureLemonadeProxySupervisorCallbacks(sup *supervisor) {
	sup.onCrash, sup.onRecovered = b.supervisedWorkerCallbacks("lemonade-proxy", func() {
		b.setLemonadeProxy(nil)
		b.unregisterService(noderec.ServiceLemonade)
	})
}

func (b *Broker) spawnLemonadeProxy() (supervisedHandle, error) {
	args := b.clusterDirArgs()
	// An adopted lemond normally owns its documented :13305. Preserve it and
	// put PAIR's facade on the adjacent port instead of competing for it.
	if !tcpPortAvailable(defaultLemonadePort) {
		args = append([]string{"--port", "13306", "--ignore-persisted-port"}, args...)
	}
	pp, err := startProxy(
		"lemonade-proxy",
		b.lemonadeProxyPath,
		applog.LevelString(),
		b.relayDir,
		b.forwardLemonadeProxyNotification,
		args...,
	)
	if err != nil {
		return nil, err
	}
	b.setLemonadeProxy(pp)
	slog.Info("lemonade-proxy started", "path", b.lemonadeProxyPath, "pid", pp.cmd.Process.Pid)
	return pp, nil
}

func (b *Broker) forwardLemonadeProxyNotification(method string, params json.RawMessage) {
	if b.dispatchErrorsNotif("lemonade-proxy", method, params) {
		return
	}
	if proxyWorkloadMethods[method] {
		b.routeProxyWorkload(method, params)
		return
	}
	if method == noderec.NotifyNodeActivity {
		b.routeNodeActivity(params)
		return
	}
	if method == "ready" {
		go b.reconcileAdvertiseLemonade(&http.Client{})
	}
	b.proxyMu.Lock()
	subscribed := b.lemonadeProxySubscribed
	b.proxyMu.Unlock()
	if !subscribed {
		return
	}
	if err := b.codec.Notify("lemonade-proxy:"+method, params); err != nil {
		slog.Warn("forward lemonade-proxy notification failed", "method", method, "err", err)
	}
}

func (b *Broker) relayToLemonadeProxy(msg *Message) {
	method := strings.TrimPrefix(msg.Method, "lemonade-proxy:")
	if method == "shutdown" {
		if err := b.codec.RespondError(msg.ID, -32601, "lemonade-proxy:shutdown is not allowed; the broker owns the proxy lifecycle"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	p := b.getLemonadeProxy()
	if p == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "lemonade-proxy not available"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	result, rpcErr, err := p.Call(context.Background(), method, msg.Params)
	switch {
	case err != nil:
		if replyErr := b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("lemonade-proxy call failed: %v", err)); replyErr != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, replyErr)
		}
	case rpcErr != nil:
		if replyErr := b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message); replyErr != nil {
			log.Printf("failed to relay lemonade-proxy error for %s: %v", msg.Method, replyErr)
		}
	default:
		if replyErr := b.codec.Respond(msg.ID, result); replyErr != nil {
			log.Printf("failed to relay lemonade-proxy result for %s: %v", msg.Method, replyErr)
		}
	}
}
