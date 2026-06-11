# NPU Migration Strategy — Frigate on Ryzen 8600G XDNA

> **Status:** Planning — hardware not yet acquired
> **Target:** Migrate Frigate object detection from ROCm/MIGraphX on Vega 8 to XDNA NPU on Ryzen 5 8600G (Phoenix)
> **Created:** 2026-06-11

---

## 1. Hardware Context

### Current (3500U Picasso)
| Resource | Detail | Limitation |
|----------|--------|------------|
| GPU | Vega 8 (gfx909) | Not ROCm-supported, GPU hangs on Liquorix 7.0.12+ |
| NPU | None | — |
| CPU | 4C/8T Zen+ | Underpowered for 6-camera detection |
| Video decode | VCN 1.0, 1-2 VAAPI sessions | Exhausted by 6 cameras |

### Target (8600G Phoenix)
| Resource | Detail | Capability |
|----------|--------|------------|
| GPU | Radeon 760M (gfx1103, RDNA3) | ROCm tier-1 supported, no hacks needed |
| NPU | XDNA 1 (npu1, PCI 1022:1502) | 10 TOPS, inference-only, kernel driver in 6.14+ |
| CPU | 6C/12T Zen 4 | ~2-3× faster per core than 3500U |
| Video decode | VCN 4.0 | Multiple simultaneous VAAPI sessions |

---

## 2. Why NPU for Frigate?

### Problem
The Vega 8 GPU requires `HSA_OVERRIDE_GFX_VERSION=10.3.0` to trick ROCm into generating gfx1030 kernels. This works but is brittle — kernel updates, ROCm updates, or driver regressions can break it (as seen with Liquorix 7.0.12).

### Opportunity
The XDNA NPU is a dedicated AI accelerator. Offloading Frigate detection to it:
- **Frees the GPU** for other AI workloads (LLMs, image generation, Open WebUI)
- **Eliminates ROCm compatibility hacks** — no more `HSA_OVERRIDE_GFX_VERSION`
- **Lower power** — NPU inference is more energy-efficient than GPU compute
- **Predictable performance** — dedicated hardware, no contention with display/decoder

### Reality Check
- XDNA 1 (Phoenix) NPU support on Linux is still maturing
- The stack: amdxdna driver → XRT → xdna-driver plugin → ONNX Runtime Vitis AI EP → model
- Issue [#686](https://github.com/amd/xdna-driver/issues/686) (Aug 2025) showed `xrt-smi examine` returning "0 devices found" on 8600G — unresolved
- The GPU works day one; the NPU is the stretch goal

---

## 3. Migration Strategy (Phased)

### Phase 0: Hardware Acquisition
- [ ] Purchase Ryzen 5 8600G (or 8700G if budget allows)
- [ ] Install CPU, update BIOS if needed
- [ ] Boot existing Linux install (Liquorix kernel 6.14+ already has amdxdna)

### Phase 1: GPU Detection First (Day 1)
- [ ] Remove `HSA_OVERRIDE_GFX_VERSION=10.3.0` from docker-compose
- [ ] Verify Radeon 760M is visible: `rocminfo`
- [ ] Verify MIGraphX model compiles for gfx1103
- [ ] Run Frigate with current ROCm detector → confirm stable
- **This is the fallback position. Frigate works. GPU hangs are gone.**

### Phase 2: NPU Driver Stack (Week 1)
- [ ] Verify kernel detects NPU: `lspci | grep IPU`, `ls /dev/accel/accel0`
- [ ] Build XRT from source (`Xilinx/XRT` master)
- [ ] Build xdna-driver plugin from `amd/xdna-driver`
- [ ] Run `xrt-smi examine` — confirm "NPU Phoenix" appears
- [ ] Run `xrt-smi validate` — confirm test workload passes
- **Decision gate:** If the device doesn't enumerate, try newer firmware, kernel parameters, or wait for driver updates. Do NOT block on this.

### Phase 3: ONNX Runtime Vitis AI (Week 2+)
- [ ] Build ONNX Runtime with Vitis AI execution provider
- [ ] Compile YOLOv8n model for XDNA NPU
- [ ] Run standalone inference benchmark — compare latency vs GPU
- [ ] Integrate into Frigate's detector framework (`xdnapu.py` plugin)

### Phase 4: Production Switch
- [ ] Run dual-detection (GPU + NPU in parallel, compare results)
- [ ] Switch primary detection to NPU once accuracy/latency validated
- [ ] GPU remains available for other AI workloads

---

## 4. Fallback Plan

If the NPU stack doesn't mature enough for production use:
1. **GPU detection** — Radeon 760M runs Frigate MIGraphX natively, stable, supported
2. **CPU detection** — Zen 4 handles YOLOv8n at 5fps × 6 cameras easily via ONNX Runtime
3. **Hybrid** — 1-2 cameras on GPU, rest on CPU, GPU free for LLMs

The 8600G upgrade is worth it regardless of NPU outcome.

---

## 5. Success Criteria

| Criterion | Target |
|-----------|--------|
| NPU driver enumerated | `xrt-smi examine` shows device |
| Model compiles for NPU | YOLOv8n ONNX → Vitis AI compiled model |
| Inference latency | < 50ms per 640×360 frame |
| Detection accuracy | Within 2% mAP of GPU baseline |
| 6-camera throughput | Stable at 5fps all cameras |
| GPU free for other work | < 10% GPU utilization during detection |

---

## 6. References

- [amd/xdna-driver](https://github.com/amd/xdna-driver) — Kernel driver + XRT plugin
- [Xilinx/XRT](https://github.com/Xilinx/XRT) — Xilinx Runtime (XRT)
- [AMD NPU on Arch Wiki](https://wiki.archlinux.org/title/AMD_NPU) — Community setup guide
- [ONNX Runtime Vitis AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html) — Official docs
- [IREE AMD AIE Plugin](https://github.com/nod-ai/iree-amd-aie) — Alternative MLIR-based path
- [Issue #686](https://github.com/amd/xdna-driver/issues/686) — 8600G enumeration failure (Aug 2025, unresolved)
