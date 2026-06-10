# Frigate NVR — Zektopic Knowledge Base

> **Purpose:** Documentation, analysis, and optimization notes for the Frigate NVR codebase.
> **Repository:** `/home/manu/Documents/github-orgs/Zektopic/frigate`
> **Last Updated:** 2026-06-10

---

## Documents

### [Code Review Issues](code-review-issues.md)
Comprehensive audit of the Frigate Python backend (~364 files). Covers:
- **3 Critical issues** — JWT security, stale detection data race
- **4 High-severity issues** — silent f-string errors, pickle safety, DB race conditions
- **7 Medium-severity issues** — hardcoded paths, exception handling gaps, restart edge cases
- **6 Low-severity issues** — memory leaks, code quality, portability

### [Optimization Guide for Low-End Hardware & AMD APUs](optimization-guide.md)
Complete configuration and tuning guide for running Frigate on:
- AMD Ryzen 5 3500U / Radeon Vega 8 Mobile (Picasso/Raven 2)
- Any AMD APU with VCN 1.0+ (pre-ROCm, no GPU compute)
- 8-16 GB RAM systems with integrated graphics
- Systems where CPU-only detection is required

Covers detection pipeline, VAAPI acceleration, memory tuning, storage optimization, CPU process tuning, and code-level improvements.

---

## Hardware Snapshot (2026-06-10)
| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 5 3500U (Zen+, 4C/8T, 2.1 GHz base) |
| GPU | Radeon Vega 8 Mobile (Picasso/Raven 2, VCN 1.0) |
| RAM | 14 GB DDR4 (shared with GPU) |
| Storage | 468 GB NVMe SSD |
| Kernel | 7.0.5-1-liquorix-amd64 |
| GPU Driver | amdgpu (in-tree) |
| ROCm | NOT supported (GFX909) |
| VAAPI | Supported via mesa-va-drivers |

---

## Key Findings

### For this hardware:
1. **CPU detection with TFLite/XNNPACK** is the only viable option — ROCm doesn't support Picasso APUs
2. **VAAPI hardware decoding** is the single biggest optimization — saves 15-25% CPU per camera
3. **640×360 @ 5 FPS** detection resolution is the sweet spot
4. **Disable all heavy features** — birdseye, audio, semantic search, face recognition, LPR, GenAI
5. **2-4 cameras max** at these settings before CPU saturation

### Code issues requiring attention:
1. **Fix the 3 silent f-string errors** in `frigate/comms/dispatcher.py` (lines 594, 608, 626) — replace bare f-strings with `logger.error()`
2. **Fix `UnboundLocalError`** in `frigate/api/auth.py:257-260` — add `continue` in the `except ValueError` block
3. **Add `try/except` around `shm.unlink()`** in `frigate/app.py:733` to prevent cleanup cascade failures

---

## Related Links
- [Frigate Documentation](https://docs.frigate.video)
- [Frigate GitHub](https://github.com/blakeblackshear/frigate)
- [VAAPI on Arch Wiki](https://wiki.archlinux.org/title/Hardware_video_acceleration)
- [AMD Picasso APU Specs](https://www.amd.com/en/products/apu/amd-ryzen-5-3500u)
