# Troubleshooting

Common issues and fixes for the ncnn Vulkan detector with Rust acceleration.

## Current Setup (2026-06-26)

**Model**: YOLO26n (ncnn Vulkan, Rust binary)  
**Architecture**: Rust detector binary (`frigate-detector-rs`) + Python ncnn subprocess  
**Speed**: ~94ms inference, ~2.8 detection FPS per camera  
**Branch**: `perf/cpu-ram-optimization`

### Architecture

```
Frigate Python (detector plugin)
  │
  ├── pipe(stdin) → frame (float16, 2.5MB)
  │
  ▼
┌─────────────────────────────────────────┐
│ frigate-detector-rs (Rust, 535KB)       │
│                                         │
│  crossbeam channels + Arc<Frame>        │
│  Python ncnn subprocess (GPU forward)   │
│  rayon-parallel argmax + NMS            │
│  sorted descending output               │
└─────────────────────────────────────────┘
  │
  ├── pipe(stdout) → detections (480B, 20×6 float32)
  │
  ▼
Frigate Python (event logic)
```

### Why Rust binary instead of inline Python worker

The inline Python worker had:
- numpy argmax/NMS in Python (70% of CPU)
- Pipe I/O overhead
- Fragile inline code (f-strings in f-strings)

The Rust binary:
- Rayon-parallel argmax + NMS (zero numpy)
- Same pipe protocol (drop-in compatible)
- Proper binary (compiled, tested)
- Architecture auto-detection (YOLO26n, YOLO11n)

### Available models

| Model | Size | Path | Status |
|-------|------|------|--------|
| **YOLO26n** | 9.3MB | `yolo26n.param` | **Active** |
| YOLOv5s | 14MB | `yolov5s.ncnn.param` | Fallback (in-process, 72ms) |
| YOLO11n | 10.2MB | `yolo11n.param` | Exported, class mapping issues |

### Quick verification
```bash
# Check model loaded
docker logs frigate | grep "model loaded\|Rust detector ready"

# Check inference speed
curl -s http://localhost:5000/api/stats | jq '.detectors."ncnn_vulkan".inference_speed'

# Verify Rust binary active
docker exec frigate ls -lh /opt/frigate/frigate-detector-rs

# Verify no errors
docker logs frigate | grep -c "short read\|buffer size\|ValueError"
# Should be 0
```

## No detections / No events

### 1. Container just restarted
Motion detectors need 2-5 minutes to calibrate. Detection FPS ramps up as cameras stabilize.

### 2. min_area too high
Check per-camera `objects.filters.person.min_area`. If the detected bounding box is smaller, it's filtered out. Default was 1000, lowered to 500.

### 3. Detection format bugs (fixed)
- X/Y swap in Rust output: `[cls,score,y1,x1,...]` vs Frigate's `[cls,score,x1,y1,...]`
- Row-major layout: ncnn output is (84 rows × 8400 cols), not column-major
- Unsorted output: Frigate's early-break expects descending scores
- Pipe partial read: `os.read` can return partial data, need loop

### 4. Record enabled: false
Recording was disabled in config. Must be `record.enabled: true`.

### 5. Offline cameras
`back_garden` at 192.168.1.20:554 is unreachable. Detection disabled for this camera.

## Rust engines deployed

| Engine | Size | Status |
|--------|------|--------|
| `frigate-detector-rs` | 535KB binary | Active (YOLO26n/11n) |
| `libfrigate_motion_rs.so` | 284KB | Built, 10/10 tests |
| `libfrigate_yolo_rs.so` | 284KB | Deployed |
| `libfrigate_frame_rs.so` | 278KB | Deployed |

## GPU info

```
GPU 0: AMD Radeon Graphics (RADV GFX1103_R1)
  Vulkan compute: ✅ (queueC=1)
  FP16 arithmetic: ✅
  INT8 compute: ❌ (int8-cm=0)
  INT8 storage: ✅
```

INT8 quantization saves VRAM but not speed. FP16 is optimal.

## Switching models
```yaml
# YOLO26n (current)
model:
  path: /config/model_cache/yolo26n.param

# YOLOv5s (in-process, no Rust binary)
model:
  path: /config/model_cache/yolov5s.ncnn.param
```

## Exporting new models to NCNN
```bash
pip install --break-system-packages ultralytics ncnn pnnx
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='ncnn', imgsz=640)
"
```
