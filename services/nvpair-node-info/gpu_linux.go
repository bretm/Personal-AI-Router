// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jaypipes/ghw"
)

// linuxGPUCommandRunner is the private seam around vendor tools. Production
// gives it a three-second, context-aware exec implementation; tests provide
// deterministic command output without requiring a GPU or either tool.
type linuxGPUCommandRunner func(context.Context, string, ...string) (string, error)

const linuxGPUCommandTimeout = 3 * time.Second

func runLinuxGPUCommand(ctx context.Context, name string, args ...string) (string, error) {
	out, err := exec.CommandContext(ctx, name, args...).Output()
	return string(out), err
}

// linuxGPUAdapter is deliberately private: PAIR has two real Linux vendor
// sources, while callers need only the existing detectGPUs and statsCollector
// interfaces. Each adapter owns its vendor-specific command and parsing.
type linuxGPUAdapter interface {
	inventory() []GPUInfo
	sample() (map[string]gpuStat, int)
}

type linuxGPUCollectors struct {
	adapters    []linuxGPUAdapter
	memoryTotal func() uint64
	fallback    func() []GPUInfo
}

func newLinuxGPUCollectors(run linuxGPUCommandRunner, memoryTotal func() uint64, fallback func() []GPUInfo) *linuxGPUCollectors {
	return &linuxGPUCollectors{
		adapters: []linuxGPUAdapter{
			&nvidiaLinuxGPUAdapter{run: run},
			&amdLinuxGPUAdapter{run: run},
		},
		memoryTotal: memoryTotal,
		fallback:    fallback,
	}
}

func defaultLinuxGPUCollectors() *linuxGPUCollectors {
	return newLinuxGPUCollectors(runLinuxGPUCommand, detectMemoryTotal, detectGPUsGHW)
}

// detectGPUs combines the inventories that NVIDIA and AMD tools can identify.
// We retain ghw only when neither vendor source has a usable inventory, which
// preserves the established generic-name fallback without duplicating adapters
// already reported by a vendor tool.
func detectGPUs() []GPUInfo {
	return defaultLinuxGPUCollectors().inventory()
}

func (c *linuxGPUCollectors) inventory() []GPUInfo {
	var gpus []GPUInfo
	for _, adapter := range c.adapters {
		gpus = appendUniqueGPUs(gpus, adapter.inventory())
	}
	if len(gpus) == 0 {
		return c.fallback()
	}
	for i := range gpus {
		if gpus[i].usesSystemMemoryUsage && gpus[i].VramBytes == 0 {
			gpus[i].VramBytes = c.memoryTotal()
		}
	}
	return gpus
}

func (c *linuxGPUCollectors) sample() (map[string]gpuStat, int) {
	stats := make(map[string]gpuStat)
	utilizationSamples := 0
	for _, adapter := range c.adapters {
		adapterStats, samples := adapter.sample()
		for key, stat := range adapterStats {
			stats[key] = stat
		}
		utilizationSamples += samples
	}
	return stats, utilizationSamples
}

func appendUniqueGPUs(dst, src []GPUInfo) []GPUInfo {
	known := make(map[string]bool, len(dst))
	for _, gpu := range dst {
		if gpu.statsKey != "" {
			known[gpu.statsKey] = true
		}
	}
	for _, gpu := range src {
		if gpu.statsKey == "" || !known[gpu.statsKey] {
			dst = append(dst, gpu)
			if gpu.statsKey != "" {
				known[gpu.statsKey] = true
			}
		}
	}
	return dst
}

// detectGPUsGHW is the generic name-only fallback, used only when neither
// vendor-specific adapter returned an inventory.
func detectGPUsGHW() []GPUInfo {
	gpu, err := ghw.GPU()
	if err != nil {
		log.Printf("GPU detection error: %v", err)
		return nil
	}
	var gpus []GPUInfo
	for _, card := range gpu.GraphicsCards {
		name := "Unknown"
		if card.DeviceInfo != nil && card.DeviceInfo.Product != nil {
			name = card.DeviceInfo.Product.Name
		}
		gpus = append(gpus, GPUInfo{Name: name})
	}
	return gpus
}

type nvidiaLinuxGPUAdapter struct {
	run         linuxGPUCommandRunner
	unavailable atomic.Bool
}

func (a *nvidiaLinuxGPUAdapter) inventory() []GPUInfo {
	csv, err := a.csv("uuid,name,memory.total")
	if err != nil {
		return nil
	}
	gpus, _ := parseNvidiaStatic(csv)
	return gpus
}

func (a *nvidiaLinuxGPUAdapter) sample() (map[string]gpuStat, int) {
	if a.unavailable.Load() {
		return nil, 0
	}
	csv, err := a.csv("uuid,utilization.gpu,memory.used")
	if err != nil {
		if a.unavailable.CompareAndSwap(false, true) {
			slog.Warn("nvidia-smi unavailable; NVIDIA GPU utilization / dedicated VRAM-used will not be reported", "err", err)
		}
		return nil, 0
	}
	return parseNvidiaDynamic(csv)
}

func (a *nvidiaLinuxGPUAdapter) csv(fields string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), linuxGPUCommandTimeout)
	defer cancel()
	return a.run(ctx, "nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits")
}

// nvidiaSmiCSV remains the production helper used by existing call sites and
// preserves its command line exactly. New Linux collection routes through the
// injected runner in nvidiaLinuxGPUAdapter.
func nvidiaSmiCSV(fields string) (string, error) {
	return (&nvidiaLinuxGPUAdapter{run: runLinuxGPUCommand}).csv(fields)
}

// isNvidiaSmiNA reports whether an nvidia-smi CSV field is a "not applicable"
// sentinel rather than a numeric value. UMA platforms such as DGX Spark return
// [N/A] or [Not Supported] for GPU memory queries.
func isNvidiaSmiNA(s string) bool {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "[]")
	switch strings.ToLower(s) {
	case "n/a", "not supported":
		return true
	default:
		return false
	}
}

// parseNvidiaStatic decodes the static query (uuid,name,memory.total) into
// GPUInfo records. memory.total is reported in MiB (because of -nounits); we
// convert to bytes. Unified-memory rows are marked so response assembly can
// source their used bytes from system memory.
func parseNvidiaStatic(out string) ([]GPUInfo, bool) {
	var gpus []GPUInfo
	var unifiedMemory bool
	for _, line := range strings.Split(out, "\n") {
		fields := splitCSVRow(line)
		if len(fields) < 3 {
			continue
		}
		uuid, name := fields[0], fields[1]
		if uuid == "" || name == "" {
			continue
		}
		var vramBytes uint64
		usesUnifiedMemory := isNvidiaSmiNA(fields[2])
		if usesUnifiedMemory {
			unifiedMemory = true
		} else if mib, err := strconv.ParseUint(fields[2], 10, 64); err == nil {
			vramBytes = mib * 1024 * 1024
		}
		gpus = append(gpus, GPUInfo{
			Name:                  name,
			VramBytes:             vramBytes,
			statsKey:              uuid,
			usesSystemMemoryUsage: usesUnifiedMemory,
		})
	}
	return gpus, unifiedMemory
}

// splitCSVRow splits one nvidia-smi CSV row on commas and trims surrounding
// whitespace from each field (the tool emits ", " separators). Returns nil for
// a blank line so callers can skip it.
func splitCSVRow(line string) []string {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

type amdLinuxGPUAdapter struct {
	run              linuxGPUCommandRunner
	unavailable      atomic.Bool
	memoryKindsMu    sync.RWMutex
	memoryKindsKnown bool
	gttMemory        map[string]bool
}

func (a *amdLinuxGPUAdapter) inventory() []GPUInfo {
	staticJSON, err := a.command("static", "--asic", "--vram", "--json")
	if err != nil {
		return nil
	}
	gpus := parseAmdStatic(staticJSON)
	if len(gpus) == 0 {
		return nil
	}
	a.setGTTMemoryKinds(gpus)
	memoryJSON, err := a.command("metric", "--mem-usage", "--json")
	if err == nil {
		applyAmdMemoryInventory(gpus, parseAmdMemory(memoryJSON))
	}
	return gpus
}

func (a *amdLinuxGPUAdapter) sample() (map[string]gpuStat, int) {
	if a.unavailable.Load() {
		return nil, 0
	}
	metricsJSON, err := a.command("metric", "--usage", "--mem-usage", "--json")
	if err != nil {
		if a.unavailable.CompareAndSwap(false, true) {
			slog.Warn("amd-smi unavailable; AMD GPU utilization / memory-used will not be reported", "err", err)
		}
		return nil, 0
	}
	return parseAmdDynamic(metricsJSON, a.gttMemoryKinds())
}

func (a *amdLinuxGPUAdapter) command(args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), linuxGPUCommandTimeout)
	defer cancel()
	return a.run(ctx, "amd-smi", args...)
}

func (a *amdLinuxGPUAdapter) setGTTMemoryKinds(gpus []GPUInfo) {
	gttMemory := make(map[string]bool, len(gpus))
	for _, gpu := range gpus {
		gttMemory[gpu.statsKey] = gpu.usesGTTMemory
	}
	a.memoryKindsMu.Lock()
	a.gttMemory = gttMemory
	a.memoryKindsKnown = true
	a.memoryKindsMu.Unlock()
}

// gttMemoryKinds lazily loads static memory kinds for the collector instance.
// Startup detection uses a separate adapter instance, so this avoids assuming
// that its process-local cache is available to the ticker.
func (a *amdLinuxGPUAdapter) gttMemoryKinds() map[string]bool {
	a.memoryKindsMu.RLock()
	if a.memoryKindsKnown {
		kinds := copyGTTMemoryKinds(a.gttMemory)
		a.memoryKindsMu.RUnlock()
		return kinds
	}
	a.memoryKindsMu.RUnlock()

	staticJSON, err := a.command("static", "--asic", "--vram", "--json")
	var gpus []GPUInfo
	if err == nil {
		gpus = parseAmdStatic(staticJSON)
	}
	a.setGTTMemoryKinds(gpus)
	return a.gttMemoryKinds()
}

func copyGTTMemoryKinds(source map[string]bool) map[string]bool {
	copy := make(map[string]bool, len(source))
	for key, value := range source {
		copy[key] = value
	}
	return copy
}

type amdSMIResponse struct {
	GPUData []amdSMIGPU `json:"gpu_data"`
}

type amdSMIGPU struct {
	GPU      int             `json:"gpu"`
	ASIC     json.RawMessage `json:"asic"`
	VRAM     json.RawMessage `json:"vram"`
	Usage    json.RawMessage `json:"usage"`
	MemUsage json.RawMessage `json:"mem_usage"`
}

type amdMemory struct {
	totalVRAM uint64
	usedVRAM  uint64
	totalGTT  uint64
	usedGTT   uint64
}

func parseAmdStatic(out string) []GPUInfo {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil
	}
	gpus := make([]GPUInfo, 0, len(response.GPUData))
	for _, gpu := range response.GPUData {
		name := amdSMIString(amdSMIField(gpu.ASIC, "market_name"))
		if name == "" || strings.EqualFold(name, "n/a") {
			continue
		}
		vramBytes, _ := amdSMIBytes(amdSMIField(gpu.VRAM, "size"))
		gpus = append(gpus, GPUInfo{
			Name:          name,
			VramBytes:     vramBytes,
			statsKey:      amdStatsKey(gpu.GPU),
			usesGTTMemory: amdSMIUsesGTTMemory(gpu.VRAM),
		})
	}
	return gpus
}

func parseAmdMemory(out string) map[string]amdMemory {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil
	}
	memory := make(map[string]amdMemory, len(response.GPUData))
	for _, gpu := range response.GPUData {
		memory[amdStatsKey(gpu.GPU)] = amdMemoryFromJSON(gpu.MemUsage)
	}
	return memory
}

func applyAmdMemoryInventory(gpus []GPUInfo, memory map[string]amdMemory) {
	for i := range gpus {
		reading, ok := memory[gpus[i].statsKey]
		if !ok {
			continue
		}
		if gpus[i].usesGTTMemory && reading.totalGTT > 0 {
			gpus[i].VramBytes = reading.totalGTT
		} else if gpus[i].VramBytes == 0 {
			gpus[i].VramBytes = reading.totalVRAM
		}
	}
}

func parseAmdDynamic(out string, gttMemory map[string]bool) (map[string]gpuStat, int) {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil, 0
	}
	stats := make(map[string]gpuStat, len(response.GPUData))
	utilizationSamples := 0
	for _, gpu := range response.GPUData {
		stat := gpuStat{}
		if utilization, ok := amdSMIPercent(amdSMIField(gpu.Usage, "gfx_activity")); ok {
			stat.UtilizationPct = utilization
			utilizationSamples++
		}
		memory := amdMemoryFromJSON(gpu.MemUsage)
		if gttMemory[amdStatsKey(gpu.GPU)] && memory.totalGTT > 0 {
			stat.VRAMUsed = memory.usedGTT
		} else {
			stat.VRAMUsed = memory.usedVRAM
		}
		stats[amdStatsKey(gpu.GPU)] = stat
	}
	return stats, utilizationSamples
}

func amdMemoryFromJSON(raw json.RawMessage) amdMemory {
	return amdMemory{
		totalVRAM: amdSMIBytesOrZero(amdSMIField(raw, "total_vram")),
		usedVRAM:  amdSMIBytesOrZero(amdSMIField(raw, "used_vram")),
		totalGTT:  amdSMIBytesOrZero(amdSMIField(raw, "total_gtt")),
		usedGTT:   amdSMIBytesOrZero(amdSMIField(raw, "used_gtt")),
	}
}

// amdSMIUsesGTTMemory distinguishes integrated DDR/LPDDR graphics memory from
// GDDR/HBM on discrete GPUs. A discrete Radeon can also expose a large GTT
// aperture, so capacity must not be inferred from GTT size alone.
func amdSMIUsesGTTMemory(vram json.RawMessage) bool {
	memoryType := strings.ToLower(amdSMIString(amdSMIField(vram, "type")))
	return strings.Contains(memoryType, "ddr") && !strings.Contains(memoryType, "gddr")
}

func parseAmdSMIResponse(out string) (amdSMIResponse, bool) {
	var response amdSMIResponse
	if err := json.Unmarshal([]byte(out), &response); err != nil {
		return amdSMIResponse{}, false
	}
	return response, true
}

func amdStatsKey(gpu int) string {
	return fmt.Sprintf("amd:%d", gpu)
}

func amdSMIField(raw json.RawMessage, name string) json.RawMessage {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil
	}
	return fields[name]
}

func amdSMIString(raw json.RawMessage) string {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

func amdSMIBytesOrZero(raw json.RawMessage) uint64 {
	bytes, _ := amdSMIBytes(raw)
	return bytes
}

// amdSMIBytes accepts the value/unit objects emitted by AMD SMI (for example
// {"value":32704,"unit":"MB"}). AMD SMI gained units after its first JSON
// releases, so a bare numeric value remains accepted for older installations.
func amdSMIBytes(raw json.RawMessage) (uint64, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, false
	}
	var measurement struct {
		Value json.RawMessage `json:"value"`
		Unit  string          `json:"unit"`
	}
	value := raw
	if err := json.Unmarshal(raw, &measurement); err == nil && len(measurement.Value) > 0 {
		value = measurement.Value
	}
	var number json.Number
	if err := json.Unmarshal(value, &number); err != nil {
		return 0, false
	}
	amount, err := strconv.ParseUint(number.String(), 10, 64)
	if err != nil {
		return 0, false
	}
	switch strings.ToUpper(strings.TrimSpace(measurement.Unit)) {
	case "", "B":
		return amount, true
	case "KB", "KIB":
		return amount * 1024, true
	case "MB", "MIB":
		return amount * 1024 * 1024, true
	case "GB", "GIB":
		return amount * 1024 * 1024 * 1024, true
	case "TB", "TIB":
		return amount * 1024 * 1024 * 1024 * 1024, true
	default:
		return 0, false
	}
}

func amdSMIPercent(raw json.RawMessage) (uint32, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, false
	}
	var measurement struct {
		Value json.RawMessage `json:"value"`
	}
	value := raw
	if err := json.Unmarshal(raw, &measurement); err == nil && len(measurement.Value) > 0 {
		value = measurement.Value
	}
	var number json.Number
	if err := json.Unmarshal(value, &number); err != nil {
		return 0, false
	}
	pct, err := strconv.ParseUint(number.String(), 10, 32)
	if err != nil {
		return 0, false
	}
	if pct > 100 {
		pct = 100
	}
	return uint32(pct), true
}
