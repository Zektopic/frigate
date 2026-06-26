# Troubleshooting

Common issues and fixes for the ncnn Vulkan detector with Rust acceleration.

## Current Setup (2026-06-27)

**Model**: YOLO26n (ncnn Vulkan, Rust binary worker)  
**Architecture**: Rust detector binary (`frigate-detector-rs`) + Python ncnn subprocess  
**Speed**: ~94ms inference, ~2.7 detection FPS per camera  
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

### Known Issue: Events UI Not Updating

**Symptom**: Recordings save to disk, detection works, but Frigate UI "Events" tab shows no new events.

**Root Cause**: Frigate's internal ZMQ event proxy (XSUB/XPUB on `ipc:///tmp/cache/proxy_{pub,sub}`) fails to initialize on this system. Affects clean upstream `ghcr.io/blakeblackshear/frigate:stable` image too — not caused by our patches.

**Workaround**: Use Frigate UI "Recordings" view (time-based) instead of "Events" tab. Recordings are saved to `/media/frigate/recordings/` every 10 seconds.

### Quick verification
```bash
# Check model loaded
docker logs frigate | grep "model loaded\|Rust detector ready"

# Check inference speed
curl -s http://localhost:5000/api/stats | jq '.detectors."ncnn_vulkan".inference_speed'

# Verify Rust binary active
docker exec frigate ls -lh /opt/frigate/frigate-detector-rs

# Verify recordings on disk
docker exec frigate ls /media/frigate/recordings/$(date +%Y-%m-%d)/*/ | head -5
```

## No detections / No events

### 1. Container just restarted
Motion detectors need 2-5 minutes to calibrate. Detection FPS ramps up as cameras stabilize.

### 2. min_area too high
Check per-camera `objects.filters.person.min_area`. Default was 1000, lowered to 500.

### 3. Detection format bugs (FIXED)
- X/Y swap in Rust output: `[cls,score,y1,x1,...]` vs Frigate's `[cls,score,x1,y1,...]`
- Row-major layout: ncnn output is (84 rows × 8400 cols), not column-major
- Unsorted output: Frigate's early-break expects descending scores
- Pipe partial read: `os.read` can return partial data, need loop

### 4. Camera offline
`back_garden` at 192.168.1.20:554 is unreachable. Detection disabled for this camera.

### 5. Record enabled: false
Recording was disabled in config. Must be `record.enabled: true`.

## Rust engines deployed

| Engine | Size | Status |
|--------|------|--------|
| `frigate-detector-rs` | 535KB binary | Active (YOLO26n/11n worker) |
| `libfrigate_motion_rs.so` | 284KB | Built, 10/10 tests, deployed |
| `libfrigate_yolo_rs.so` | 284KB | Built, yolo26_post_process active |
| `libfrigate_frame_rs.so` | 278KB | Built, YUV ops deployed |

## Detection pipeline bugs fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| X/Y swap | Boxes had swapped coordinates | Output `[cls,score,x1,y1,x2,y2]` |
| Row-major layout | Scores = 637 (garbage) | ncnn output is (84×8400) row-major |
| Unsorted output | First detection low → all skipped | Sort by score descending |
| Pipe partial read | ValueError crash | Loop os.read until complete |
| Mask inversion | motion-rs zeroed all pixels | mask[i]==0 means UNMASKED |

## Switching models
```yaml
# YOLO26n (current)
model:
  path: /config/model_cache/yolo26n.param
  width: 640
  height: 640

# YOLOv5s (in-process, 72ms, more false positives)
model:
  path: /config/model_cache/yolov5s.ncnn.param
```

## GPU info
```
GPU 0: AMD Radeon Graphics (RADV GFX1103_R1)
  Vulkan compute: ✅ (queueC=1)
  FP16 arithmetic: ✅
  INT8 compute: ❌ (int8-cm=0)
```
FP16 is optimal. INT8 saves VRAM but not speed.

## Exporting new models to NCNN
```bash
pip install --break-system-packages ultralytics ncnn pnnx
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='ncnn', imgsz=640)
"
```
