# Frigate NCNN Vulkan GPU Detector

Drop-in replacement for Frigate's ONNX detector that uses **ncnn with Vulkan backend** instead of ONNX Runtime with ROCm. This bypasses AMD's unstable ROCm compute stack on Linux by using the battle-tested **RADV** (Mesa Vulkan) driver.

## Why?

AMD's ROCm compute stack has broken GPU reset on iGPUs (Phoenix1/Radeon 760M, gfx1103). MIGraphX hangs the GPU within 6 minutes. The RADV Vulkan driver is stable (it's what every Linux game uses), and ncnn provides a production-grade Vulkan inference backend.

## Supported Models

| Architecture | Models | Output Format | Vulkan | Status |
|---|---|---|---|---|
| Anchor-based (YOLOv5/v7) | YOLOv5s, YOLOv7-tiny | 3 outputs × 255 ch | ✅ Yes | **YOLOv5s: stable** |
| Anchor-free (YOLOv8/11) | YOLOv8n, YOLO11n | 1 output × 144 ch (DFL) | ⚠️ In progress | Segfaults in Frigate worker process |

The plugin **auto-detects** model architecture from the `.param` file and picks the correct post-processing pipeline (sigmoid + anchors for YOLOv5/v7, DFL decode for YOLOv8/YOLO11). Input/output blob names are also auto-detected — no hardcoded names.

### Downloading models

All models from [nihui/ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models):

```bash
cd config/model_cache/

# YOLOv5s — stable, recommended
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.param
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.bin

# YOLOv7-tiny — experimental (may have GPU-specific issues)
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov7-tiny.bin
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov7-tiny.param

# YOLOv8n — anchor-free, DFL decode (segfaults in forked process, under investigation)
# Already included in model_cache

# YOLO11n — newest architecture (segfaults in forked process, under investigation)
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolo11n.ncnn.param
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolo11n.ncnn.bin
```

## Quick Start

### 1. Build the custom image

```bash
docker build -t frigate-ncnn-vulkan docker/ncnn-vulkan/
```

### 2. Configure Frigate

```yaml
detectors:
  ncnn_vulkan:
    type: onnx
    device: GPU

model:
  model_type: yolo-generic
  width: 640
  height: 640
  input_tensor: nchw
  input_dtype: float
  path: /config/model_cache/yolov5s.ncnn.param
  labelmap_path: /labelmap/coco-80.txt
```

Use `type: onnx` — the plugin replaces the ONNX detector internally.

**Important:** Set `model.path` to the `.param` file of your chosen model. The plugin auto-detects the matching `.bin` file.

### 3. Run

```bash
docker compose up -d
```

## How It Works

```
YOLO model .param/.bin → ncnn (Vulkan EP) → SPIR-V shaders → RADV → GPU
                                    ↑
                           bypasses ROCm entirely
```

The plugin:
1. Reads the `.param` file to detect architecture (anchor-based vs anchor-free)
2. Auto-detects input/output blob names (handles `in0`, `images`, `out0`, ONNX-style names, etc.)
3. Selects the appropriate post-processing: sigmoid + Frigate's `post_process_yolo` for anchor-based models, or custom DFL decode + NMS for anchor-free models
4. Rescales detections to the original frame size

## Verified Hardware

| GPU | Driver | Status |
|---|---|---|
| AMD Radeon 760M (gfx1103) | RADV GFX1103_R1 | ✅ Stable |
| Any GPU with Vulkan 1.2+ | Mesa RADV | ✅ Expected to work |

## Performance (Radeon 760M, YOLOv5s 640×640)

- Inference: ~121ms (Frigate pipeline, 5 cameras)
- CPU usage: ~65% (detector process)
- No GPU hangs or resets needed
