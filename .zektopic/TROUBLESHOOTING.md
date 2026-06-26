# Troubleshooting

Common issues and fixes for the ncnn Vulkan detector.

## Current Setup (2026-06-26)

**Model**: YOLOv5s (ncnn Vulkan, in-process)  
**Architecture**: In-process ncnn Vulkan (no worker subprocess)  
**Speed**: ~75ms inference, ~2.1-2.9 detection FPS per camera  
**Branch**: `perf/cpu-ram-optimization`

### Why YOLOv5s instead of YOLO26n

YOLO26n (and all anchor-free models: YOLOv8, YOLOv9, YOLO11) crash with SIGSEGV when running Vulkan in-process after Frigate's fork(). This is an AMD RADV Vulkan bug affecting the DFL grid-decode code path on Radeon 760M (gfx1103).

The worker subprocess workaround was tried but caused:
- 223% CPU overhead from pipe IPC + Python subprocess
- Fragile communication (broken pipes, silent failures)
- 94ms inference (slower than in-process 75ms)

YOLOv5s (anchor-based) uses a different Vulkan code path that is stable after fork. Runs at 75ms in-process with zero worker overhead.

### Quick verification
```bash
# Check model loaded correctly
docker logs frigate | grep "model loaded"
# Should show: NCNN: model loaded (arch=yolov5, Vulkan=on)

# Check inference speed
curl -s http://localhost:5000/api/stats | jq '.detectors."ncnn_vulkan".inference_speed'
# Should be ~75ms (YOLOv5s in-process)

# Verify NO worker subprocess
docker exec frigate ps aux | grep "python3 -c" | grep -c ncnn
# Should be 0

# Check for segfaults
docker logs frigate | grep -c "segfault\|Fatal"
# Should be 0
```

## No detections (events not triggering)

### Symptom
Detection FPS shows in stats but no events are generated.

### Common causes

1. **Container just restarted** — Motion detectors need 30-60 seconds to calibrate. Detection FPS will ramp up as cameras stabilize.

2. **score_thresh too high** — Worker uses 0.05 internally. Camera `min_score` is 0.2 in config.

3. **Frigate detect() threshold** — Default is 0.2 (lowered from 0.4). Patched in `object_detection/base.py`.

4. **Model output format mismatch** — YOLOv5s outputs 3 scales × 255 channels. The plugin's `_detect_architecture` auto-detects from `.param` file.

## Switching between YOLOv5s and YOLO26n

### YOLOv5s (current — recommended)
```yaml
model:
  path: /config/model_cache/yolov5s.ncnn.param
  width: 640
  height: 640
```
- In-process Vulkan (no worker)
- 75ms inference
- Stable, no crashes
- More false positives but reliable

### YOLO26n (faster, more accurate, but needs worker)
```yaml
model:
  path: /config/model_cache/yolo26n.param
  width: 640
  height: 640
```
- Worker subprocess required (pipe IPC)
- ~94ms inference with worker
- Higher CPU usage (~223% worker process)
- Fewer false positives
- Fragile pipe communication

## GPU not detected by ncnn

### Symptom
```
NCNN: no GPU found, using CPU fallback
```

### Fix
1. Verify Mesa Vulkan drivers: `apt install mesa-vulkan-drivers`
2. Check GPU visible: `docker exec frigate python3 -c "import ncnn; print(ncnn.get_gpu_count())"`
3. Check `/dev/dri/renderD128` passthrough in docker-compose
4. Verify render group access: `ls -la /dev/dri/renderD128` (should be `root:render`, gid 992)

## Detection is slow (>200ms)

1. Verify Vulkan: `docker logs frigate | grep "Vulkan=on"`
2. Check no worker process: `docker exec frigate ps aux | grep "python3 -c"`
3. Verify model is YOLOv5s (in-process path)

## Model cache cleanup

Only keep working models:
```bash
ls /mnt/docker/frigate/config/model_cache/
# Keep: yolov5s.ncnn.param, yolov5s.ncnn.bin
# Keep: yolo26n.param, yolo26n.bin (for future use)
# Remove: yolov8*, yolov9*, yolo11*, yolov7-tiny* (all crash on this GPU)
```

## Address already in use (OSError 98)

Harmless — Frigate's internal auth service port conflict during startup. Retries and starts correctly.

## Rust SIMD engines

Three Rust `.so` engines are deployed at `/opt/frigate/`:
- `libfrigate_motion_rs.so` — Motion detection (mask bug fixed, 10/10 tests)
- `libfrigate_yolo_rs.so` — YOLO post-processing (YOLO26 + anchor-free + NMS)
- `libfrigate_frame_rs.so` — Frame preprocessing (YUV ops, resize, normalize)

Verify they're loadable:
```bash
docker exec frigate python3 -c "
from frigate.motion.rust_engine import motion_available
from frigate.detectors.rust_yolo import yolo_available
print(f'motion_rs: {motion_available()}')
print(f'yolo_rs: {yolo_available()}')
"
```
