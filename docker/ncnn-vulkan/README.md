# Frigate NCNN Vulkan GPU Detector

Drop-in replacement for Frigate's ONNX detector that uses **ncnn with Vulkan backend** instead of ONNX Runtime with ROCm. This bypasses AMD's unstable ROCm compute stack on Linux by using the battle-tested **RADV** (Mesa Vulkan) driver.

## Why?

AMD's ROCm compute stack has broken GPU reset on iGPUs (Phoenix1/Radeon 760M, gfx1103). MIGraphX hangs the GPU within 6 minutes. The RADV Vulkan driver is stable (it's what every Linux game uses), and ncnn provides a production-grade Vulkan inference backend.

## How It Works

```
YOLOv5s ONNX → ncnn (Vulkan EP) → SPIR-V shaders → RADV → GPU
                        ↑
               bypasses ROCm entirely
```

## Quick Start

### 1. Build the custom image

```bash
docker build -t frigate-ncnn-vulkan docker/ncnn-vulkan/
```

### 2. Download the ncnn model

```bash
cd config/model_cache/
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.param
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.bin
```

### 3. Configure Frigate

```yaml
detectors:
  ncnn_vulkan:
    type: onnx
    device: GPU
```

Use `type: onnx` — the plugin replaces the ONNX detector internally.

### 4. Run

```bash
docker compose up -d
```

## Verified Hardware

| GPU | Driver | Status |
|---|---|---|
| AMD Radeon 760M (gfx1103) | RADV GFX1103_R1 | ✅ Stable |
| Any GPU with Vulkan 1.2+ | Mesa RADV | ✅ Expected to work |

## Performance (Radeon 760M)

- Inference: ~28ms (YOLOv5s 640x640)
- CPU usage: ~20%
- No GPU hangs or resets needed
