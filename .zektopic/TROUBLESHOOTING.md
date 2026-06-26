# Troubleshooting

Common issues and fixes for the ncnn Vulkan detector.

## Current Setup (2026-06-26)

**Model**: YOLO26n (Ultralytics export, ncnn Vulkan)  
**Architecture**: Worker subprocess (clean GPU context, pipe IPC with float16)  
**Speed**: ~94ms inference, ~11 FPS  
**Branch**: `fix/yolo26-events`

### Quick verification
```bash
# Check model loaded correctly
docker logs frigate | grep "model loaded"
# Should show: NCNN: Vulkan inference worker ready

# Check inference speed
curl -s http://localhost:5000/api/stats | jq '.detectors."ncnn_vulkan".inference_speed'
# Should be ~90-100ms

# Check for segfaults
docker logs frigate | grep -c "segfault\|Fatal"
# Should be 0
```

### How the worker subprocess works
```
Frigate detector (forked)          Worker subprocess (clean fork)
        │                                │
        ├── frame (float16, 2.5MB) ──→ pipe ──→ ncnn Vulkan GPU (~88ms)
        │                                │
        │                                ├── process_yolo26() numpy (~5ms)
        │                                │     - cx,cy,w,h → x1,y1,x2,y2
        │                                │     - NMS (top 20 detections)
        │                                │
        │ ←── 20 detections (480B) ──── pipe ───┤
        │
        └── Frigate event pipeline
```

The worker is needed because YOLO26 (like YOLOv8/v9/v11) crashes with SIGSEGV when running Vulkan in-process after Frigate's fork(). This is an AMD RADV Vulkan bug affecting anchor-free models on Radeon 760M (gfx1103). The clean subprocess has no inherited GPU state and works fine.

Float16 pipe: frames are sent as float16 (2.5MB) instead of float32 (5MB) — halves pipe I/O CPU usage.

## No detections (events not triggering)

### Symptom
Detection FPS shows in stats but no events are generated.

### Worker diagnostics
```bash
# Check worker is alive
docker exec frigate ps aux | grep "python3 -c" | head -1
# Worker PID should exist with high CPU (~240%)

# Check for broken pipe errors
docker logs frigate | grep "Broken pipe"
# Zero broken pipes = worker is healthy

# Check for segfaults
docker logs frigate | grep -c "segfault\|Fatal"
# Must be 0
```

### Common causes

1. **score_thresh too high** — in worker code, `process_yolo26` uses `score_thresh=0.1`. If increased, fewer detections pass to Frigate.

2. **Frigate min_score too high** — per-camera `min_score` filters worker output. Current thresholds:
   - Most cameras: person 0.55, car 0.55
   - Back garden/kitchen: person 0.60

3. **Worker crash on startup** — check `docker logs frigate | grep "NCNN worker: pipe error"`. If present, worker is crashing. Common causes:
   - Syntax error in f-string worker code (check for missing `result = dets.tobytes()`)
   - Model file missing/corrupt in `/config/model_cache/yolo26n.*`

4. **Model output format mismatch** — YOLO26 outputs (84, 8400). If the plugin's `_detect_architecture` doesn't detect `"yolo26"`, it falls back to wrong processing.

## Switching back to YOLOv5s (fallback)

If YOLO26 has issues, switch back to the stable YOLOv5s:
```yaml
model:
  path: /config/model_cache/yolov5s.ncnn.param
  width: 640
  height: 640
```
Then `docker compose restart frigate`.

YOLOv5s runs in-process (no worker) at ~140ms, 7 FPS. More false positives but rock-solid stable.

## Exporting new YOLO models to NCNN

```bash
# Inside Frigate container or any environment with ultralytics + ncnn + pnnx
pip install --break-system-packages ultralytics ncnn pnnx
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')  # or yolo26s.pt for small variant
model.export(format='ncnn', imgsz=640)
"
# Output: yolo26n_ncnn_model/model.ncnn.param + model.ncnn.bin
```

Copy to model cache and update config path.

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

## Address already in use (OSError 98)

Harmless — Frigate's internal auth service port conflict during startup. Retries and starts correctly.

## Detection is slow (>200ms)

1. Verify Vulkan: `docker logs frigate | grep "Vulkan=on"`
2. Check worker alive: `docker exec frigate ps aux | grep "python3 -c"`
3. Check for segfaults causing worker restarts

## Model cache cleanup

Only keep working models to avoid confusion:
```bash
ls /mnt/docker/frigate/config/model_cache/
# Should have: yolov5s.ncnn.param, yolov5s.ncnn.bin, yolo26n.param, yolo26n.bin
# Remove: yolov8*, yolov9*, yolo11*, yolov7-tiny* (all crash on this GPU)
```
