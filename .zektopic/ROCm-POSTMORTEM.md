# ROCm on Radeon 760M — Postmortem

## The Problem

AMD's ROCm compute stack consistently hangs the Radeon 760M (gfx1103, Phoenix1 APU) on Linux. All attempts were verified on:
- Kernel: 7.0.11-76070011-generic (Pop!_OS 24.04)
- ROCm: 7.2.3 (userspace)
- amdgpu kernel driver: matching 7.2.3

## What We Tried

### Attempt 1: MIGraphX + VAAPI decode
```yaml
detectors:
  onnx:
    device: GPU    # MIGraphXExecutionProvider
ffmpeg:
  hwaccel_args: preset-vaapi
```
**Result**: GPU hung after ~1 hour. TTM memory manager deadlock between VAAPI (media engine) and ROCm (compute engine).

### Attempt 2: MIGraphX only (no VAAPI)
```yaml
ffmpeg:
  hwaccel_args: ""   # software decode
```
**Result**: GPU hung after **6-9 minutes**. Proved the issue is ROCm compute itself, not VAAPI contention.

### Attempt 3: MIGraphX + stability env vars
```bash
MIGRAPHX_DISABLE_MIOPEN_FUSION=1
MIGRAPHX_DISABLE_REDUCE_FUSION=1
MIGRAPHX_DISABLE_SCHEDULE_PASS=1
MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS=1
```
**Result**: GPU hung during MODEL COMPILATION (66 seconds in). Even compilation crashes the GPU.

## Root Cause

The kernel log showed:
```
[37456.170181] amdgpu 0000:0f:00.0: GPU reset begin!. Source: 5
```

Source 5 = compute engine hang/timeout. But the reset **never completed** — no "GPU reset end" message appears. After the reset attempt, 16 TTM kernel workers were stuck in D state (uninterruptible sleep), and the container couldn't be killed.

This is a **known AMD Linux driver bug** for Phoenix1 iGPUs:
1. ROCm compute work triggers a GPU engine hang
2. The kernel attempts GPU reset
3. GPU reset on gfx1103 **fails to complete**
4. All processes accessing the GPU are stuck forever in D state
5. Only a hard reboot recovers

## Why Windows Works

Windows uses a completely different driver stack:
- **WDDM memory manager**: per-engine isolation prevents media/compute deadlocks
- **TDR (Timeout Detection & Recovery)**: GPU reset completes reliably in ~2 seconds
- **DirectML**: runs on DirectX, bypassing ROCm entirely

## Why Vulkan (RADV) Works

RADV is Mesa's community-developed Vulkan driver. Unlike AMD's proprietary ROCm stack:
- RADV has been tested by millions of Linux gamers
- GPU reset handling is mature (tested extensively via DXVK/vkd3d)
- Vulkan compute uses the same robust submit/fence infrastructure as Vulkan graphics
- No TTM-specific code paths that are buggy on iGPUs

## Key Lessons

1. **Never use ROCm on AMD iGPUs (Phoenix1, Rembrandt, etc.)** — the kernel driver's GPU reset is broken
2. **HSA_OVERRIDE_GFX_VERSION is a hack, not a fix** — it makes ROCm detect the GPU but doesn't fix the underlying driver bugs
3. **VAAPI + ROCm contention is real** — the TTM memory manager can deadlock when both engines run simultaneously
4. **RADV is the production path for AMD GPU compute on Linux** — it bypasses all ROCm kernel driver issues

## Recovery

When (not if) the GPU hangs:
1. `docker kill` doesn't work — processes in D state can't be signaled
2. Even `reboot` may hang on the GPU driver
3. Hard power cycle is the only reliable recovery
