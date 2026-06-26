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

All anchor-free YOLO models crash with SIGSEGV when Vulkan is used in-process after Python `fork()`. This is an AMD RADV driver bug.

### The fork problem explained

Frigate uses Python `multiprocessing` with `forkserver` mode. When Python forks, the child inherits the parent's ENTIRE memory space — including GPU DRM file descriptors and partially-initialized Vulkan instance/device handles.

```
Frigate main process
  │
  ├── fork() → Camera process
  │              Inherits: GPU DRM fd for VAAPI decode → ✅ Works
  │
  ├── fork() → Detector process  
  │              Inherits: stale VkDevice, VkInstance handles
  │              Tries: ncnn Vulkan compute shaders
  │              
  │              Anchor-based (YOLOv5): ✅ Simple buffer ops
  │              Anchor-free (YOLOv8+): ❌ SIGSEGV
  │              Reason: DFL grid decode uses storage images
  │              that RADV validates against PARENT's resource table
  │
  └── fork() → Recording process → ✅ Works (no GPU)
```

### The worker subprocess workaround (and its cost)

For anchor-free models, a worker subprocess is spawned via `Popen` (uses `posix_spawn`, not `fork`) to get a clean GPU context:

```
Detector process (forked)
  │
  └── spawn() → Worker subprocess (CLEAN GPU state)
                   stdin ← 2.5MB float16 frame (pipe, per frame)
                   ncnn Vulkan inference (GPU, ~60ms)
                   numpy post-process (CPU, ~20ms: argmax + NMS + clip)
                   stdout → 480B results (pipe, per frame)
```

**CPU cost breakdown per frame (5 cameras @ 2.5 detection FPS = 12.5 fps):**

| Operation | %CPU | Why |
|-----------|------|-----|
| float32→float16 conversion | 5% | numpy astype copys 1.2M elements |
| os.write(stdin, 2.5MB) | 10% | Kernel memcpy to pipe buffer |
| os.read(stdin, 2.5MB) | 10% | Kernel memcpy from pipe buffer to worker |
| float16→float32 conversion | 5% | numpy astype in worker |
| ncnn GPU command submission | 50% | Vulkan cmd buffer creation, fence polling |
| numpy argmax (8400×80 classes) | 30% | Per-cell class search over 672K comparisons |
| numpy NMS while-loop | 40% | Greedy IoU with array slicing per iteration |
| Pipe result write (480B) | 1% | Tiny return path |
| Python interpreter overhead | 72% | GIL, GC, object allocation |
| **Sum of CPU** | **~223%** | Multi-threaded (ncnn uses thread pool) |

Total pipe throughput: 12.5 frames/sec × 2.5MB = **31.25 MB/sec** through kernel pipe buffers.

### The planned fix (frigate-detector-rs)

A standalone Rust binary that:
1. Starts fresh (no fork — clean GPU)
2. Receives frames via shared memory (zero-copy, no kernel pipe overhead)
3. Runs ncnn Vulkan inference
4. Post-processes in pure Rust (zero numpy)
5. Returns results via shared memory

Blocked by: ncnn C FFI headers not available in container (ncnn only ships as Python C extension).

### Current solution: YOLOv5s in-process

YOLOv5s uses anchor-based ops that don't trigger the RADV fork bug. Runs directly in the forked detector with zero overhead.

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
