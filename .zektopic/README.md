# Zektopic Frigate Customizations

Production-grade GPU-accelerated object detection for Frigate on AMD GPUs — without ROCm.

## Overview

AMD's ROCm compute stack has broken GPU reset on iGPUs (Phoenix1/Radeon 760M). This project provides a drop-in replacement detector that uses **ncnn with Vulkan (RADV)** instead, completely bypassing ROCm while keeping full GPU acceleration.

## Contents

| Document | Description |
|---|---|
| [GPU-DETECTOR.md](GPU-DETECTOR.md) | ncnn Vulkan detector — how it works, setup, benchmarks |
| [ROCm-POSTMORTEM.md](ROCm-POSTMORTEM.md) | Why ROCm failed on Radeon 760M and what we learned |
| [CONFIG.md](CONFIG.md) | Complete Frigate configuration reference |
| [BUILD-GUIDE.md](BUILD-GUIDE.md) | How to build the custom Docker image |
| [RUST-MIGRATION.md](RUST-MIGRATION.md) | Python to Rust split, migration status, and future roadmap |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and fixes |

## Quick Results

| Metric | CPU-only | ROCm/MIGraphX | ncnn Vulkan (this) |
|---|---|---|---|
| Inference speed | 37-77ms | 10ms | 28-35ms |
| CPU usage | 97% | 26% | **14%** |
| GPU usage | 0% | 40% (hung) | **40% (stable)** |
| Stability | ✅ | ❌ 6-min hangs | ✅ |
| Detection FPS | 2-5 | crashed | 7-8 per camera |

## Key Finding

The Radeon 760M's GPU reset mechanism is broken in the amdgpu kernel driver on Linux (source 5 = compute engine hang → reset never completes). This affects ALL ROCm workloads. The fix is to use RADV (Mesa Vulkan driver) via ncnn instead of ROCm.
