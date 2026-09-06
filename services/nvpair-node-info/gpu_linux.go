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
	return newLinuxGPUCollectorsWithAMDDRM(run, memoryTotal, fallback, func() []amdDRMGPU { return nil })
}

func newLinuxGPUCollectorsWithAMDDRM(run linuxGPUCommandRunner, memoryTotal func() uint64, fallback func() []GPUInfo, readDRM amdDRMReader) *linuxGPUCollectors {
	return &linuxGPUCollectors{
		adapters: []linuxGPUAdapter{
			&nvidiaLinuxGPUAdapter{run: run},
			&amdLinuxGPUAdapter{run: run, readDRM: readDRM, fallback: fallback},
		},
		memoryTotal: memoryTotal,
		fallback:    fallback,
	}
}

func defaultLinuxGPUCollectors() *linuxGPUCollectors {
	return newLinuxGPUCollectorsWithAMDDRM(runLinuxGPUCommand, detectMemoryTotal, detectGPUsGHW, readAMDDRMGPUs)
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
		gpus = append(gpus, GPUInfo{Name: name, pciAddress: normalizePCIBDF(card.Address)})
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
	run         linuxGPUCommandRunner
	readDRM     amdDRMReader
	fallback    func() []GPUInfo
	unavailable atomic.Bool
	contextMu   sync.RWMutex
	contextSet  bool
	context     amdGPUContext
}

type amdGPUContext struct {
	statsKeys map[int]string
	gttMemory map[string]bool
}

func (a *amdLinuxGPUAdapter) inventory() []GPUInfo {
	drm := a.drmGPUs()
	staticJSON, err := a.command("static", "--asic", "--bus", "--vram", "--json")
	records := parseAmdStaticRecords(staticJSON)
	if err != nil || len(records) == 0 {
		context := amdContextFromDRM(drm)
		a.setGPUContext(context)
		return amdDRMInventory(drm, a.fallbackInventory())
	}
	reconcileAmdStaticWithDRM(records, drm)
	context := amdContextFromStatic(records)
	a.setGPUContext(context)
	gpus := amdGPUInfos(records)
	memoryJSON, err := a.command("metric", "--mem-usage", "--json")
	if err == nil {
		applyAmdMemoryInventory(gpus, parseAmdMemoryWithKeys(memoryJSON, context.statsKeys))
	}
	applyAmdDRMInventory(gpus, drm)
	return gpus
}

func (a *amdLinuxGPUAdapter) sample() (map[string]gpuStat, int) {
	drm := a.drmGPUs()
	context := a.gpuContext(drm)
	stats := make(map[string]gpuStat)
	utilizationKeys := make(map[string]bool)
	if !a.unavailable.Load() {
		metricsJSON, err := a.command("metric", "--usage", "--mem-usage", "--json")
		if err != nil {
			if a.unavailable.CompareAndSwap(false, true) {
				slog.Warn("amd-smi unavailable; using kernel DRM telemetry for AMD GPUs", "err", err)
			}
		} else {
			stats, utilizationKeys = parseAmdDynamicWithKeys(metricsJSON, context.gttMemory, context.statsKeys)
		}
	}
	if stats == nil {
		stats = make(map[string]gpuStat)
	}
	if utilizationKeys == nil {
		utilizationKeys = make(map[string]bool)
	}
	mergeAmdDRMStats(stats, utilizationKeys, drm, context)
	return stats, len(utilizationKeys)
}

func (a *amdLinuxGPUAdapter) command(args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), linuxGPUCommandTimeout)
	defer cancel()
	return a.run(ctx, "amd-smi", args...)
}

func (a *amdLinuxGPUAdapter) drmGPUs() []amdDRMGPU {
	if a.readDRM == nil {
		return nil
	}
	return a.readDRM()
}

func (a *amdLinuxGPUAdapter) fallbackInventory() []GPUInfo {
	if a.fallback == nil {
		return nil
	}
	return a.fallback()
}

// gpuContext lazily establishes the AMD-SMI index-to-PCI mapping for the
// independently created background collector. If AMD-SMI is unavailable,
// DRM order supplies a deterministic mapping and the kernel remains usable.
func (a *amdLinuxGPUAdapter) gpuContext(drm []amdDRMGPU) amdGPUContext {
	a.contextMu.RLock()
	if a.contextSet {
		context := copyAmdGPUContext(a.context)
		a.contextMu.RUnlock()
		return context
	}
	a.contextMu.RUnlock()

	staticJSON, err := a.command("static", "--asic", "--bus", "--vram", "--json")
	records := parseAmdStaticRecords(staticJSON)
	var context amdGPUContext
	if err == nil && len(records) > 0 {
		reconcileAmdStaticWithDRM(records, drm)
		context = amdContextFromStatic(records)
	} else {
		context = amdContextFromDRM(drm)
	}
	a.setGPUContext(context)
	return copyAmdGPUContext(context)
}

func (a *amdLinuxGPUAdapter) setGPUContext(context amdGPUContext) {
	a.contextMu.Lock()
	a.context = copyAmdGPUContext(context)
	a.contextSet = true
	a.contextMu.Unlock()
}

func copyAmdGPUContext(source amdGPUContext) amdGPUContext {
	copy := amdGPUContext{
		statsKeys: make(map[int]string, len(source.statsKeys)),
		gttMemory: make(map[string]bool, len(source.gttMemory)),
	}
	for index, key := range source.statsKeys {
		copy.statsKeys[index] = key
	}
	for key, value := range source.gttMemory {
		copy.gttMemory[key] = value
	}
	return copy
}

type amdSMIResponse struct {
	GPUData []amdSMIGPU `json:"gpu_data"`
}

type amdSMIGPU struct {
	GPU      int             `json:"gpu"`
	ASIC     json.RawMessage `json:"asic"`
	Bus      json.RawMessage `json:"bus"`
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

type amdStaticRecord struct {
	index int
	bdf   string
	info  GPUInfo
}

func parseAmdStaticRecords(out string) []amdStaticRecord {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil
	}
	records := make([]amdStaticRecord, 0, len(response.GPUData))
	for _, gpu := range response.GPUData {
		name := amdSMIString(amdSMIField(gpu.ASIC, "market_name"))
		if name == "" || strings.EqualFold(name, "n/a") {
			continue
		}
		bdf := normalizePCIBDF(amdSMIString(amdSMIField(gpu.Bus, "bdf")))
		vramBytes, _ := amdSMIBytes(amdSMIField(gpu.VRAM, "size"))
		records = append(records, amdStaticRecord{
			index: gpu.GPU,
			bdf:   bdf,
			info: GPUInfo{
				Name:          name,
				VramBytes:     vramBytes,
				statsKey:      amdStatsKeyForBDF(gpu.GPU, bdf),
				pciAddress:    bdf,
				usesGTTMemory: amdSMIUsesGTTMemory(gpu.VRAM),
			},
		})
	}
	return records
}

func parseAmdStatic(out string) []GPUInfo {
	return amdGPUInfos(parseAmdStaticRecords(out))
}

func amdGPUInfos(records []amdStaticRecord) []GPUInfo {
	gpus := make([]GPUInfo, 0, len(records))
	for _, record := range records {
		gpus = append(gpus, record.info)
	}
	return gpus
}

func reconcileAmdStaticWithDRM(records []amdStaticRecord, drm []amdDRMGPU) {
	byBDF := make(map[string]amdDRMGPU, len(drm))
	for _, gpu := range drm {
		byBDF[gpu.bdf] = gpu
	}
	for i := range records {
		device, ok := byBDF[records[i].bdf]
		if !ok && records[i].bdf == "" && i < len(drm) {
			device = drm[i]
			ok = true
		}
		if !ok {
			continue
		}
		records[i].bdf = device.bdf
		records[i].info.pciAddress = device.bdf
		records[i].info.statsKey = amdStatsKeyForBDF(records[i].index, device.bdf)
		if device.usesUnifiedMemory {
			records[i].info.usesGTTMemory = true
			if device.gttTotalKnown {
				records[i].info.VramBytes = device.gttTotal
			}
		} else if records[i].info.VramBytes == 0 && device.vramTotalKnown {
			records[i].info.VramBytes = device.vramTotal
		}
	}
}

func amdContextFromStatic(records []amdStaticRecord) amdGPUContext {
	context := amdGPUContext{
		statsKeys: make(map[int]string, len(records)),
		gttMemory: make(map[string]bool, len(records)),
	}
	for _, record := range records {
		context.statsKeys[record.index] = record.info.statsKey
		context.gttMemory[record.info.statsKey] = record.info.usesGTTMemory
	}
	return context
}

func amdContextFromDRM(drm []amdDRMGPU) amdGPUContext {
	context := amdGPUContext{
		statsKeys: make(map[int]string, len(drm)),
		gttMemory: make(map[string]bool, len(drm)),
	}
	for index, gpu := range drm {
		key := amdStatsKeyForBDF(index, gpu.bdf)
		context.statsKeys[index] = key
		context.gttMemory[key] = gpu.usesUnifiedMemory
	}
	return context
}

func amdDRMInventory(drm []amdDRMGPU, fallback []GPUInfo) []GPUInfo {
	names := make(map[string]string, len(fallback))
	for _, gpu := range fallback {
		if gpu.pciAddress != "" {
			names[normalizePCIBDF(gpu.pciAddress)] = gpu.Name
		}
	}
	gpus := make([]GPUInfo, 0, len(drm))
	for index, device := range drm {
		name := names[device.bdf]
		if name == "" {
			name = unknownAMDDeviceName
			if device.deviceID != "" {
				name += " 0x" + device.deviceID
			}
		}
		gpu := GPUInfo{
			Name:          name,
			statsKey:      amdStatsKeyForBDF(index, device.bdf),
			pciAddress:    device.bdf,
			usesGTTMemory: device.usesUnifiedMemory,
		}
		if device.usesUnifiedMemory && device.gttTotalKnown {
			gpu.VramBytes = device.gttTotal
		} else if device.vramTotalKnown {
			gpu.VramBytes = device.vramTotal
		}
		gpus = append(gpus, gpu)
	}
	return gpus
}

func applyAmdDRMInventory(gpus []GPUInfo, drm []amdDRMGPU) {
	byBDF := make(map[string]amdDRMGPU, len(drm))
	for _, device := range drm {
		byBDF[device.bdf] = device
	}
	for i := range gpus {
		device, ok := byBDF[normalizePCIBDF(gpus[i].pciAddress)]
		if !ok {
			continue
		}
		if device.usesUnifiedMemory && device.gttTotalKnown {
			gpus[i].VramBytes = device.gttTotal
		} else if !device.usesUnifiedMemory && device.vramTotalKnown {
			gpus[i].VramBytes = device.vramTotal
		}
	}
}

func mergeAmdDRMStats(stats map[string]gpuStat, utilizationKeys map[string]bool, drm []amdDRMGPU, context amdGPUContext) {
	for index, device := range drm {
		key := amdDRMStatsKey(index, device, context)
		if key == "" {
			continue
		}
		stat := stats[key]
		haveStat := false
		if device.utilizationKnown {
			stat.UtilizationPct = device.utilizationPercent
			utilizationKeys[key] = true
			haveStat = true
		}
		if device.usesUnifiedMemory && device.gttUsedKnown {
			stat.VRAMUsed = device.gttUsed
			haveStat = true
		} else if !device.usesUnifiedMemory && device.vramUsedKnown {
			stat.VRAMUsed = device.vramUsed
			haveStat = true
		}
		if haveStat {
			stats[key] = stat
		}
	}
}

func amdDRMStatsKey(index int, device amdDRMGPU, context amdGPUContext) string {
	bdfKey := amdStatsKeyForBDF(index, device.bdf)
	for _, key := range context.statsKeys {
		if key == bdfKey {
			return key
		}
	}
	if key := context.statsKeys[index]; key != "" && !strings.HasPrefix(key, "amd:pci:") {
		return key
	}
	return bdfKey
}

func parseAmdMemory(out string) map[string]amdMemory {
	return parseAmdMemoryWithKeys(out, nil)
}

func parseAmdMemoryWithKeys(out string, statsKeys map[int]string) map[string]amdMemory {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil
	}
	memory := make(map[string]amdMemory, len(response.GPUData))
	for _, gpu := range response.GPUData {
		memory[amdStatsKeyFromMap(gpu.GPU, statsKeys)] = amdMemoryFromJSON(gpu.MemUsage)
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
	stats, utilizationKeys := parseAmdDynamicWithKeys(out, gttMemory, nil)
	return stats, len(utilizationKeys)
}

func parseAmdDynamicWithKeys(out string, gttMemory map[string]bool, statsKeys map[int]string) (map[string]gpuStat, map[string]bool) {
	response, ok := parseAmdSMIResponse(out)
	if !ok {
		return nil, nil
	}
	stats := make(map[string]gpuStat, len(response.GPUData))
	utilizationKeys := make(map[string]bool)
	for _, gpu := range response.GPUData {
		key := amdStatsKeyFromMap(gpu.GPU, statsKeys)
		stat := gpuStat{}
		if utilization, ok := amdSMIPercent(amdSMIField(gpu.Usage, "gfx_activity")); ok {
			stat.UtilizationPct = utilization
			utilizationKeys[key] = true
		}
		memory := amdMemoryFromJSON(gpu.MemUsage)
		if gttMemory[key] && memory.totalGTT > 0 {
			stat.VRAMUsed = memory.usedGTT
		} else {
			stat.VRAMUsed = memory.usedVRAM
		}
		stats[key] = stat
	}
	return stats, utilizationKeys
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

func amdStatsKeyForBDF(gpu int, bdf string) string {
	if bdf = normalizePCIBDF(bdf); bdf != "" {
		return "amd:pci:" + bdf
	}
	return amdStatsKey(gpu)
}

func amdStatsKeyFromMap(gpu int, statsKeys map[int]string) string {
	if key := statsKeys[gpu]; key != "" {
		return key
	}
	return amdStatsKey(gpu)
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
