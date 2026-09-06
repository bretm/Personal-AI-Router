// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	defaultDRMClassPath  = "/sys/class/drm"
	defaultKFDNodesPath  = "/sys/class/kfd/kfd/topology/nodes"
	amdPCIVendorID       = 0x1002
	unknownAMDDeviceName = "AMD GPU"
)

// amdDRMGPU is the kernel's view of one AMD DRM adapter. The PCI BDF is the
// stable join key shared with AMD-SMI. Individual counters remain optional so
// an older kernel can still contribute the fields it does expose.
type amdDRMGPU struct {
	bdf                string
	deviceID           string
	renderMinor        int
	usesUnifiedMemory  bool
	vramTotal          uint64
	vramTotalKnown     bool
	vramUsed           uint64
	vramUsedKnown      bool
	gttTotal           uint64
	gttTotalKnown      bool
	gttUsed            uint64
	gttUsedKnown       bool
	utilizationPercent uint32
	utilizationKnown   bool
}

type amdDRMReader func() []amdDRMGPU

func readAMDDRMGPUs() []amdDRMGPU {
	return readAMDDRMGPUsAt(defaultDRMClassPath, defaultKFDNodesPath)
}

func readAMDDRMGPUsAt(drmRoot, kfdNodesRoot string) []amdDRMGPU {
	unifiedRenderMinors := readUnifiedAMDRenderMinors(kfdNodesRoot)
	entries, err := os.ReadDir(drmRoot)
	if err != nil {
		return nil
	}

	var gpus []amdDRMGPU
	for _, entry := range entries {
		if !isDRMCardName(entry.Name()) {
			continue
		}
		deviceRoot := filepath.Join(drmRoot, entry.Name(), "device")
		vendor, ok := readUintFile(filepath.Join(deviceRoot, "vendor"), 0)
		if !ok || vendor != amdPCIVendorID {
			continue
		}

		uevent, _ := os.ReadFile(filepath.Join(deviceRoot, "uevent"))
		fields := parseKeyValueLines(string(uevent))
		bdf := normalizePCIBDF(fields["PCI_SLOT_NAME"])
		if bdf == "" {
			continue
		}

		gpu := amdDRMGPU{
			bdf:         bdf,
			deviceID:    normalizePCIID(fields["PCI_ID"]),
			renderMinor: readDRMRenderMinor(filepath.Join(deviceRoot, "drm")),
		}
		gpu.usesUnifiedMemory = unifiedRenderMinors[gpu.renderMinor]
		gpu.vramTotal, gpu.vramTotalKnown = readUintFile(filepath.Join(deviceRoot, "mem_info_vram_total"), 10)
		gpu.vramUsed, gpu.vramUsedKnown = readUintFile(filepath.Join(deviceRoot, "mem_info_vram_used"), 10)
		gpu.gttTotal, gpu.gttTotalKnown = readUintFile(filepath.Join(deviceRoot, "mem_info_gtt_total"), 10)
		gpu.gttUsed, gpu.gttUsedKnown = readUintFile(filepath.Join(deviceRoot, "mem_info_gtt_used"), 10)
		if utilization, valid := readUintFile(filepath.Join(deviceRoot, "gpu_busy_percent"), 10); valid {
			if utilization > 100 {
				utilization = 100
			}
			gpu.utilizationPercent = uint32(utilization)
			gpu.utilizationKnown = true
		}
		gpus = append(gpus, gpu)
	}

	sort.Slice(gpus, func(i, j int) bool {
		return gpus[i].bdf < gpus[j].bdf
	})
	return gpus
}

func isDRMCardName(name string) bool {
	if !strings.HasPrefix(name, "card") || len(name) == len("card") {
		return false
	}
	_, err := strconv.ParseUint(strings.TrimPrefix(name, "card"), 10, 32)
	return err == nil
}

func readDRMRenderMinor(drmRoot string) int {
	entries, err := os.ReadDir(drmRoot)
	if err != nil {
		return -1
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "renderD") {
			continue
		}
		minor, err := strconv.Atoi(strings.TrimPrefix(entry.Name(), "renderD"))
		if err == nil {
			return minor
		}
	}
	return -1
}

// readUnifiedAMDRenderMinors uses KFD's local_mem_size as the authoritative
// distinction between dedicated VRAM and an APU's system-memory pool. A GPU
// node with SIMD units and zero local memory is unified-memory; this avoids
// trusting AMD-SMI versions that render an unknown VRAM enum as "GDDR7".
func readUnifiedAMDRenderMinors(root string) map[int]bool {
	result := make(map[int]bool)
	entries, err := os.ReadDir(root)
	if err != nil {
		return result
	}
	for _, entry := range entries {
		properties, err := os.ReadFile(filepath.Join(root, entry.Name(), "properties"))
		if err != nil {
			continue
		}
		fields := parseSpaceSeparatedProperties(string(properties))
		renderMinor, haveRenderMinor := fields["drm_render_minor"]
		localMemory, haveLocalMemory := fields["local_mem_size"]
		simdCount, haveSIMDCount := fields["simd_count"]
		if haveRenderMinor && haveLocalMemory && haveSIMDCount && simdCount > 0 && localMemory == 0 {
			result[int(renderMinor)] = true
		}
	}
	return result
}

func parseKeyValueLines(input string) map[string]string {
	result := make(map[string]string)
	for _, line := range strings.Split(input, "\n") {
		key, value, ok := strings.Cut(line, "=")
		if ok {
			result[strings.TrimSpace(key)] = strings.TrimSpace(value)
		}
	}
	return result
}

func parseSpaceSeparatedProperties(input string) map[string]uint64 {
	result := make(map[string]uint64)
	for _, line := range strings.Split(input, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err == nil {
			result[fields[0]] = value
		}
	}
	return result
}

func readUintFile(path string, base int) (uint64, bool) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	value, err := strconv.ParseUint(strings.TrimSpace(string(contents)), base, 64)
	return value, err == nil
}

func normalizePCIBDF(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func normalizePCIID(value string) string {
	parts := strings.Split(strings.TrimSpace(value), ":")
	if len(parts) != 2 {
		return ""
	}
	return strings.ToLower(parts[1])
}
