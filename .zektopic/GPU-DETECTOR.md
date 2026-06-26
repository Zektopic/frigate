# NCNN Vulkan GPU Detector

Drop-in replacement for Frigate's ONNX detector using ncnn with Vulkan backend via RADV.

## Architecture (Current: YOLOv5s in-process)

```
Camera frame → Frigate preprocessing → ncnn Vulkan → RADV → GPU
                                                         ↑
                                              (no ROCm involved)
```

YOLOv5s uses the **in-process** ncnn Vulkan path. The model loads and runs entirely within the forked detector process — no worker subprocess, no pipe IPC. This is stable on AMD RADV because YOLOv5's anchor-based operations use a GPU code path that doesn't trigger the fork-related SIGSEGV.

## Why NOT anchor-free models (YOLOv8/YOLO11/YOLO26)

All anchor-free YOLO models crash with SIGSEGV when Vulkan is used in-process after Python fork(). This is an AMD RADV driver bug affecting the DFL (Distribution Focal Loss) grid-decode code path on Radeon 760M (gfx1103).

The only workaround is a worker subprocess with pipe IPC, which adds:
- 223% CPU overhead
- Pipe I/O latency (2.5MB per frame)
- Fragile communication
- Debugging complexity

## Setup

### 1. Build the custom image

```bash
cd docker/ncnn-vulkan
docker build -t frigate-ncnn-vulkan .
```

### 2. Download the ncnn YOLO model

```bash
cd config/model_cache/
# YOLOv5s (in-process, recommended)
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.param
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.bin

# YOLO26n (needs worker subprocess)
# Export from ultralytics: YOLO('yolo26n.pt').export(format='ncnn')
```

### 3. Configure Frigate

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
  path: /config/model_cache/yolov5s.ncnn.param  # in-process path
  labelmap_path: /labelmap/coco-80.txt
```

### 4. docker-compose.yml

```yaml
services:
  frigate:
    image: frigate-ncnn-vulkan
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    # No /dev/kfd needed — Vulkan doesn't use ROCm
```

## How It Works

1. **Plugin discovery**: Frigate scans `detectors/plugins/` for Python modules. Our `onnx.py` registers with `type_key = "onnx"`, hijacking the ONNX detector slot.

2. **Architecture auto-detection**: The plugin reads the `.param` file to detect YOLOv5 (255 channels) vs YOLOv8/YOLO26 (other channel counts). YOLOv5 uses the in-process Vulkan path; anchor-free models spawn a worker subprocess.

3. **Model loading**: ncnn loads the model (.param + .bin) at startup. GPU is selected automatically via Vulkan device enumeration. FP16 arithmetic is enabled for speed.

4. **Inference**: Each frame arrives preprocessed from Frigate (resized to 640×640, RGB, NCHW format, normalized to [0,1]). We multiply by 255 to convert back to [0,255] for ncnn YOLOv5s. ncnn runs the model on GPU via Vulkan compute shaders.

5. **Postprocessing**: For YOLOv5s — sigmoid on raw logits, anchor-based box decode, NMS. For YOLO26 — the model outputs decoded boxes directly, so only NMS is needed.

## Performance (Radeon 760M / Phoenix1)

| Model | Architecture | Inference | CPU | Notes |
|-------|-------------|-----------|-----|-------|
| YOLOv5s | In-process | ~75ms | ~50% total | Stable, recommended |
| YOLO26n | Worker subprocess | ~94ms | ~223% worker | Higher accuracy, more CPU |
| YOLO26n | In-process | ~30ms | ❌ SIGSEGV | Too unstable |

## Verified Hardware

| GPU | Driver | Vulkan after fork? | Status |
|-----|--------|-------------------|--------|
| AMD Radeon 760M (gfx1103) | RADV | Anchor-based: ✅, Anchor-free: ❌ | Current |
| Any Vulkan 1.2+ GPU | Mesa RADV | Depends on model type | Expected |
| Vega 8 (gfx909) | RADV | Same limitations | Expected |

## Rust SIMD Engines

Three Rust crates built as `.so` libraries for CPU acceleration:
- `frigate-motion-rs`: Motion detection pipeline (AVX2 SIMD)
- `frigate-yolo-rs`: YOLO post-processing (grid decode + NMS)
- `frigate-frame-rs`: Frame preprocessing (YUV ops, resize, normalize)

All deployed at `/opt/frigate/`. Python bindings in `frigate/motion/rust_engine.py` and `frigate/detectors/rust_yolo.py`.

## INT8 Quantization

Radeon 760M supports INT8 storage but NOT INT8 compute (`int8-cm=0`). FP16 is the optimal precision for this GPU. INT8 quantization would save VRAM but not improve inference speed.
