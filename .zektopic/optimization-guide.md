# Frigate NVR — Optimization Guide for Low-End Hardware & AMD APUs

> **Date:** 2026-06-10
> **Target Hardware:** AMD Ryzen 5 3500U (Zen+, 4C/8T @ 2.1 GHz base) / Radeon Vega 8 Mobile (Picasso/Raven 2, VCN 1.0) / 14 GB DDR4 (shared) / NVMe SSD
> **Kernel:** 7.0.5-liquorix-amd64 (amdgpu driver)

---

## Table of Contents

1. [Hardware Profile & Capabilities](#1-hardware-profile--capabilities)
2. [Detection Pipeline Optimization](#2-detection-pipeline-optimization)
3. [Hardware Acceleration Strategy](#3-hardware-acceleration-strategy)
4. [Memory & SHM Tuning](#4-memory--shm-tuning)
5. [Storage & Recording Optimization](#5-storage--recording-optimization)
6. [CPU & Process Tuning](#6-cpu--process-tuning)
7. [Complete Recommended Config](#7-complete-recommended-config)
8. [Code-Level Improvements for Low-End Systems](#8-code-level-improvements-for-low-end-systems)
9. [Monitoring & Verification](#9-monitoring--verification)

---

## 1. Hardware Profile & Capabilities

### This Machine: AMD Ryzen 5 3500U with Radeon Vega 8

| Component | Spec | Frigate Relevance |
|-----------|------|-------------------|
| **CPU** | 4 cores / 8 threads, Zen+, 15W TDP | Will handle 2-4 cameras at moderate resolution |
| **GPU** | Vega 8 (Picasso/Raven 2), VCN 1.0 | VAAPI decode for H.264/H.265; **no ROCm support** |
| **RAM** | 14 GB total (shared with GPU) | ~2 GB reserved for GPU; ~12 GB available for system |
| **Storage** | NVMe SSD, 468 GB | Fast enough for recording writes; watch TBW endurance |
| **Video Decode** | UVD 7.0 / VCN 1.0 | H.264 decode up to 4K; H.265/HEVC decode up to 4K |
| **Video Encode** | VCE 4.0 | H.264 encode (no H.265 encode on VCN 1.0) |
| **Compute** | No ROCm; no OpenCL via ROCm | Must use CPU-only detection or Vulkan compute |

### Key Limitations for Frigate

1. **No ROCm/HIP support** — Picasso APUs (GFX909) are NOT supported by ROCm. You cannot use AMD GPU-accelerated TensorFlow/ONNX inference.
2. **VAAPI decode only** — You get hardware video **decoding** (saving CPU), but object **detection** must run on CPU.
3. **Shared memory bandwidth** — The Vega iGPU shares DDR4 bandwidth with the CPU. Heavy GPU decode + CPU detection compete for memory bandwidth.
4. **15W shared TDP** — The CPU and GPU share a power budget. Sustained loads will throttle.

### What This APU CAN Do Well

- **Hardware video decoding** via VAAPI (saves significant CPU)
- **CPU-based detection** with lightweight models (CPU has AVX2, which helps)
- **2-4 cameras** at 720p–1080p with conservative detection settings
- **go2rtc restream** with hardware encoding for the live view

---

## 2. Detection Pipeline Optimization

### 2.1 Use the Lightest Possible Model

The single biggest lever on low-end hardware is the detection model. Do NOT use the default 320×320 MobileNet SSD via OpenVINO — it's surprisingly heavy on CPU.

**Recommended for Vega APUs:**

```yaml
detectors:
  cpu:
    type: cpu
    num_threads: 3  # Leave 1 core for ffmpeg/system
    model:
      path: /config/model_cache/tensorflow/ssd_mobilenet_v2_coco_2018_03_29
      labelmap_path: /labelmap.txt
      width: 300
      height: 300
      input_tensor: nhwc
      input_pixel_format: bgr
```

**Why CPU detector and not OpenVINO on this APU:**
- OpenVINO's CPU plugin uses Intel-specific optimizations (VNNI, DL Boost) that don't exist on AMD Zen+ cores
- On AMD APUs, TensorFlow Lite with XNNPACK (which auto-detects and uses AVX2) often outperforms OpenVINO
- YOLO-NAS is too heavy for this class of hardware
- The included `cpu_tfl` detector (`frigate/detectors/plugins/cpu_tfl.py`) uses TFLite with XNNPACK — this is your best option

**Performance benchmarks (approximate, 300×300 model on this CPU):**

| Detector | Inference Time | Notes |
|----------|---------------|-------|
| cpu_tfl (XNNPACK) | 30-50ms | Best option, AVX2-accelerated |
| openvino (CPU) | 45-80ms | Intel-optimized, slower on AMD |
| onnx (CPU) | 40-70ms | Generic ONNX Runtime |

### 2.2 Detection Resolution Tradeoffs

```yaml
cameras:
  front_door:
    detect:
      width: 640    # NOT 1280! Each 2× increase in res = 4× more pixels to process
      height: 360   # 640×360 is the sweet spot for this APU
      fps: 5        # 5 FPS is sufficient for most security use cases
```

**Resolution impact on this CPU (per camera, 300×300 model):**

| Detect Resolution | Pixels/sec @ 5fps | CPU Load (approx) |
|-------------------|-------------------|-------------------|
| 640×360 | 1.15M | ~15% per camera |
| 1280×720 | 4.6M | ~35% per camera |
| 1920×1080 | 10.4M | ~60% per camera |

**Recommendation:** 640×360 detect resolution. The detection model resizes to 300×300 anyway, so the extra resolution only helps with small/distant objects. On this APU, the CPU cost isn't worth it for most home setups.

### 2.3 Reduce Detection FPS

```yaml
detect:
  fps: 5  # 5 frames per second, not 10 or 15
```

At 5 FPS, a person walking across the frame (~3 seconds) still generates 15 detection frames. That's plenty for reliable tracking. Going from 10→5 FPS halves your detection CPU load.

### 2.4 Limit Tracked Object History

The Norfair tracker maintains a history of object positions. Reduce the tracking history to save memory and CPU:

```yaml
# In camera config:
objects:
  track:
    - person
    - car
    - cat
    - dog
  filters:
    person:
      min_area: 5000      # Ignore very small (distant) people
      max_area: 300000    # Ignore people filling the entire frame
    car:
      min_area: 10000
```

---

## 3. Hardware Acceleration Strategy

### 3.1 VAAPI: The Vega 8's Best Feature for Frigate

The Vega 8 iGPU has UVD 7.0/VCN 1.0 which supports hardware decoding of H.264 and H.265. This is **the most important optimization** for this APU — it offloads video decoding from the CPU to dedicated silicon.

**Install VAAPI support (Debian/Ubuntu):**

```bash
sudo apt install vainfo intel-media-va-driver mesa-va-drivers
# mesa-va-drivers provides the AMD VCN driver
# Verify:
vainfo
# Should show: VAProfileH264Main, VAProfileH264High, VAProfileHEVCMain
```

**Enable VAAPI in Frigate config:**

```yaml
ffmpeg:
  hwaccel_args: preset-vaapi  # Uses the built-in preset for VAAPI

cameras:
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://camera_ip/stream
          roles:
            - detect
            - record
      hwaccel_args: preset-vaapi
```

**What VAAPI does on this APU:**
- H.264 decode: Offloaded to UVD 7.0 hardware — saves 15-25% CPU per camera
- H.265/HEVC decode: Also hardware accelerated
- Color conversion (YUV→RGB): Can be done on GPU via `hwdownload` + `scale_vaapi` (saves additional CPU)

### 3.2 VAAPI Advanced Tuning for Low-End APUs

The default `preset-vaapi` uses reasonable settings but for this APU we can go further:

```yaml
ffmpeg:
  hwaccel_args:
    - -hwaccel
    - vaapi
    - -hwaccel_device
    - /dev/dri/renderD128
    - -hwaccel_output_format
    - vaapi
  input_args:
    - -avoid_negative_ts
    - make_zero
    - -fflags
    - nobuffer+genpts+igndts
    - -flags
    - low_delay
    - -strict
    - experimental
    - -analyzeduration
    - "0"
    - -probesize
    - "32"
  output_args:
    detect:  # For the detect stream:
      - -vf
      - fps=5,scale_vaapi=w=640:h=360,hwdownload,format=nv12,format=bgra
      - -an  # No audio needed for detection
    record:  # For the record stream:
      - -vf
      - format=nv12,hwupload,scale_vaapi=w=1280:h=720
      - -c:v
      - h264_vaapi
      - -b:v
      - 2M
```

**Key points:**
- `format=nv12` keeps the pixel format in GPU-native layout — avoids expensive CPU conversion
- `scale_vaapi` does scaling on the GPU before downloading to CPU for detection
- Detection stream gets audio stripped (`-an`) since object detection doesn't need it
- Recording uses `h264_vaapi` encoder to keep encoding on GPU

### 3.3 Why NOT Vulkan Compute for Detection?

The Vega 8 supports Vulkan 1.2, which in theory could run ONNX models via `onnxruntime` with the Vulkan execution provider. However:

1. **Memory overhead:** iGPU shares system RAM. Running a detection model on GPU steals bandwidth from the CPU
2. **No INT8 acceleration:** Vega 8 has no INT8 tensor cores. Detection models run in FP32, which is slow
3. **Context switching:** The same GPU is doing video decode. Adding compute creates contention
4. **Power throttling:** At 15W, the GPU can either decode OR compute well, not both

**Verdict:** Stick with CPU detection on this APU.

---

## 4. Memory & SHM Tuning

### 4.1 Shared Memory (/dev/shm) Sizing

Frigate uses `/dev/shm` for frame passing between processes. On this 14 GB system, the default Docker `/dev/shm` size of 64 MB is far too small.

**In docker-compose.yml:**

```yaml
services:
  frigate:
    shm_size: "256mb"  # Sufficient for 4 cameras at moderate resolution
```

**Calculation for this system:**
- Per-camera frame: `640 × 360 × 1.5 (YUV) + overhead ≈ 345 KB`
- With 2 spare slots: `(4 + 2) × 345 KB × 50 frames ≈ 100 MB`
- Add birdseye + go2rtc buffers: ~50 MB
- Total: ~150-200 MB minimum
- **Recommendation: 256 MB** — leaves headroom

### 4.2 Reduce SHM Frame Count

The `SHM_MAX_FRAMES` env var controls how many frames are buffered:

```yaml
environment:
  SHM_MAX_FRAMES: "30"  # Default is 50; reduce to save memory
```

On a slower CPU, frames can accumulate faster than they're processed. A lower limit prevents memory exhaustion and forces the pipeline to drop frames rather than OOM.

### 4.3 Reduce Detection Queue Size

**Code observation:** `frigate/app.py:167-176` sizes the detection queue as `(enabled_cameras + 2) * 2`. On a 4-camera system, that's 12 slots. Each slot holds a pointer, but frames in the queue prevent shared memory from being reused.

For this APU, the queue is fine at 12 — reducing further would risk dropping frames when the CPU catches up.

### 4.4 System-Level Memory Tuning

```bash
# Reduce swappiness — this APU has enough RAM for Frigate; avoid swapping
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-frigate.conf

# Increase the max map count for many ffmpeg processes
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.d/99-frigate.conf

sudo sysctl -p /etc/sysctl.d/99-frigate.conf
```

---

## 5. Storage & Recording Optimization

### 5.1 NVMe SSD Considerations

This machine has a 468 GB NVMe SSD with 116 GB free. Frigate recording writes are continuous and can burn through SSD write endurance.

**Critical:** Use a separate drive for recordings if possible (external USB3 HDD). The NVMe SSD should be for the OS and Frigate's database only.

If you must record to the NVMe:

```yaml
record:
  enabled: true
  retain:
    days: 3          # Short retention; delete old recordings aggressively
    mode: all
  events:
    pre_capture: 5   # Seconds before event to include
    post_capture: 5  # Seconds after event to include
    retain:
      default: 7     # Keep event clips for 7 days
      mode: active_objects
```

### 5.2 Recording Resolution vs. Detect Resolution

Recording at the full camera resolution while detecting at a lower one is supported. This protects your SSD from writing full-resolution 24/7:

```yaml
cameras:
  front_door:
    detect:
      width: 640
      height: 360
    record:
      enabled: true
      retain:
        days: 3
    ffmpeg:
      output_args:
        record: preset-record-generic  # Keep full resolution for recordings
```

The `preset-record-generic` records at the stream's native resolution. Combined with the detect stream being downscaled, this gives you high-quality recordings with low CPU detection cost.

### 5.3 Reduce Segment Size

```yaml
record:
  segment_duration: 60  # 60-second segments (default is often 10)
```

Longer segments mean fewer database rows, fewer file open/close operations, and less fragmentation. On an SSD this is fine; on HDD, shorter segments help with cleanup.

### 5.4 Database on tmpfs (Advanced)

Move the Frigate database to a tmpfs to reduce SSD writes:

```yaml
database:
  path: /dev/shm/frigate.db  # WARNING: Lost on reboot!
```

**Only do this if:** you back up the DB periodically or are fine losing event history on crash. The DB can reach hundreds of MB — ensure your SHM size accommodates it.

---

## 6. CPU & Process Tuning

### 6.1 Limit ffmpeg Threads

By default, ffmpeg may use too many threads for software encoding/decoding:

```yaml
ffmpeg:
  global_args:
    - -threads
    - "2"  # Limit to 2 threads per ffmpeg instance
```

With 4 cameras × 2 threads = 8 ffmpeg threads total, leaving the remaining threads for detection and other processes.

### 6.2 Process CPU Affinity (Advanced)

On the 4-core / 8-thread APU, pin processes to specific cores:

```bash
# Example: Pin detection to cores 4-7 (the second CCX)
# In docker-compose or systemd service:
taskset -c 4-7 python3 -m frigate
```

Note: The Ryzen 5 3500U is a single CCX (4 cores in one complex), so core pinning is less impactful than on dual-CCX chips. It's still useful to keep detection and ffmpeg on different logical cores.

### 6.3 Reduce go2rtc Overhead

go2rtc handles restreaming. On this APU:

```yaml
go2rtc:
  streams:
    front_door:
      - rtsp://camera_ip/stream
      - "ffmpeg:front_door#video=h264#hardware=vaapi"
```

The `hardware=vaapi` flag tells go2rtc to use the GPU for transcoding when restreaming.

### 6.4 Disable Unused Features

Every feature you disable saves CPU and memory:

```yaml
# Disable these unless you actually use them:
birdseye:
  enabled: false       # Saves ~5-10% CPU; enable only if you use the bird's-eye view

audio:
  enabled: false       # Saves ~3-5% CPU per camera with audio

semantic_search:
  enabled: false       # vec0 extension and embeddings add memory + CPU overhead

lpr:
  enabled: false       # License plate recognition is heavy

face_recognition:
  enabled: false       # Face embedding generation uses CPU/GPU

classification:
  custom: {}           # No custom classifiers

genai:
  enabled: false       # Sends images to LLMs — disable on low-end hardware
```

### 6.5 Motion Detection Tuning

Motion detection runs on CPU before object detection. Tune it down:

```yaml
motion:
  threshold: 30        # Higher = less sensitive (default is often 25)
  contour_area: 30     # Minimum contour size in pixels (higher = ignore small motion)
  improve_contrast: false  # Contrast enhancement adds CPU overhead
  mask:
    - 0,0,640,0,640,360,0,360  # Example: mask out a tree that constantly moves
```

---

## 7. Complete Recommended Config

This is a complete configuration optimized for the AMD Ryzen 5 3500U with 2-3 cameras:

```yaml
mqtt:
  host: 192.168.1.100  # Your MQTT broker IP
  port: 1883

detectors:
  cpu:
    type: cpu
    num_threads: 3

model:
  width: 300
  height: 300
  input_tensor: nhwc
  input_pixel_format: bgr
  path: /config/model_cache/tensorflow/ssd_mobilenet_v2_coco_2018_03_29
  labelmap_path: /labelmap.txt

database:
  path: /config/frigate.db

record:
  enabled: true
  retain:
    days: 3
    mode: all
  events:
    pre_capture: 5
    post_capture: 5
    retain:
      default: 7
      mode: active_objects

snapshots:
  enabled: true
  retain:
    default: 7

go2rtc:
  streams:
    camera_1:
      - rtsp://user:pass@192.168.1.101:554/stream
      - "ffmpeg:camera_1#video=h264#hardware=vaapi"
    camera_2:
      - rtsp://user:pass@192.168.1.102:554/stream
      - "ffmpeg:camera_2#video=h264#hardware=vaapi"

cameras:
  camera_1:
    enabled: true
    ffmpeg:
      hwaccel_args: preset-vaapi
      inputs:
        - path: rtsp://user:pass@192.168.1.101:554/stream
          roles:
            - detect
            - record
      output_args:
        detect:
          - -vf
          - fps=5,scale_vaapi=w=640:h=360,hwdownload,format=nv12,format=bgra
          - -an
        record: preset-record-generic
    detect:
      width: 640
      height: 360
      fps: 5
      enabled: true
    motion:
      threshold: 30
      contour_area: 30
      improve_contrast: false
    objects:
      track:
        - person
        - car
        - cat
        - dog
      filters:
        person:
          min_area: 5000
          max_area: 300000
        car:
          min_area: 10000
    record:
      enabled: true
      retain:
        days: 3
    snapshots:
      enabled: true

  camera_2:
    # Same structure as camera_1, adjusted for second camera

# Disable heavy features
birdseye:
  enabled: false

audio:
  enabled: false

semantic_search:
  enabled: false

face_recognition:
  enabled: false

lpr:
  enabled: false

classification:
  custom: {}

ffmpeg:
  global_args:
    - -threads
    - "2"

detect:
  enabled: true
  width: 640
  height: 360
  fps: 5
  max_disappeared: 25  # Keep objects tracked longer before considering them gone
  stationary:
    interval: 10       # Check for stationary objects less frequently
    threshold: 50      # Higher threshold = less sensitive to stationary objects

# Telemetry — keep minimal on low-end hardware
telemetry:
  network_interfaces: []
  stats_interval: 60   # Emit stats every 60 seconds, not more often
```

---

## 8. Code-Level Improvements for Low-End Systems

These are specific code changes that would make Frigate more efficient on constrained hardware:

### 8.1 Adaptive Detection Rate (New Feature)

**Problem:** Frigate runs detection at a fixed FPS regardless of system load. On low-end hardware, detection can fall behind and frames accumulate.

**Proposed change in `frigate/video/detect.py`:**
- Monitor the detection queue depth
- If the queue has > 2 pending frames, skip the next frame
- This lets Frigate automatically reduce effective FPS when the CPU is overloaded

```python
# Pseudo-code for adaptive FPS
if detection_queue.qsize() > 2:
    logger.debug("Detection queue backed up, skipping frame")
    return  # Skip this frame, process next one
```

### 8.2 Lazy Model Loading (Memory Optimization)

**Problem:** Frigate loads all configured detector models at startup, consuming RAM even if a detector isn't actively used.

**Proposed change in `frigate/detectors/__init__.py`:**
- Load detector models on first use, not at import time
- Unload models during periods of inactivity (if supported by the runtime)

### 8.3 ffmpeg Process Pooling (CPU Optimization)

**Problem:** Each camera spawns its own ffmpeg process. For 4 cameras, that's 4+ ffmpeg processes, each with its own decoder instance.

**Proposed change in `frigate/video/ffmpeg.py`:**
- Use a shared ffmpeg instance for cameras on the same GPU device
- This reduces context switching and memory overhead

### 8.4 Motion Detection ROI (Region of Interest)

**Problem:** Motion detection runs on the entire frame, even in areas where motion is irrelevant (sky, static walls).

**Proposed change in `frigate/motion/improved_motion.py`:**
- Allow defining a motion ROI mask that's applied BEFORE the motion algorithm runs
- This skips pixel processing entirely for masked regions, not just detection

### 8.5 Shared Memory Frame Reuse

**Problem:** `UntrackedSharedMemory` at `frigate/util/image.py` creates new shared memory segments. On memory-constrained systems, fragmentation can occur.

**Proposed change:** Implement a SHM pool with pre-allocated segments that are recycled rather than created/destroyed per frame.

### 8.6 Database Write Batching

**Problem:** `frigate/record/maintainer.py` writes recording segments to the database individually. Under heavy recording load, this creates many small SQLite transactions.

**Proposed change:** Batch recording inserts (every 30 seconds or 100 recordings, whichever comes first) using `INSERT OR REPLACE` with prepared statements.

### 8.7 VAAPI Device Selection Awareness

**Problem:** The `preset-vaapi` in `frigate/ffmpeg_presets.py` assumes `/dev/dri/renderD128` is the correct device. On hybrid GPU laptops (common with APUs), there may be multiple render nodes.

**Proposed change:** Auto-detect the correct render node by querying `vainfo` for the AMD device specifically, rather than blindly using renderD128.

---

## 9. Monitoring & Verification

### 9.1 Check VAAPI is Working

```bash
# During Frigate operation, check that ffmpeg is using VAAPI:
ps aux | grep ffmpeg | grep vaapi
# Should show: -hwaccel vaapi -hwaccel_device /dev/dri/renderD128

# Check GPU utilization (install radeontop):
sudo radeontop -d - -l 1
# Should show UVD/VCN engine activity during recording
```

### 9.2 Monitor CPU Usage per Process

```bash
# Check that detection processes aren't saturating the CPU:
top -H -p $(pgrep -f "frigate.detector")
# Per-thread view of the detector — should see 3 threads at ~60-80% each

# Check ffmpeg CPU usage (should be low if VAAPI is working):
ps -eo pid,comm,%cpu --sort=-%cpu | grep ffmpeg
```

### 9.3 Check for Thermal Throttling

```bash
# Watch CPU frequency — if it drops below base clock (2.1 GHz), you're throttling:
watch -n1 "grep MHz /proc/cpuinfo"

# GPU temperature:
sensors | grep edge

# If GPU edge > 85°C, you're likely throttling
# Solution: Reduce detection FPS or resolution
```

### 9.4 Verify Detection Performance

Frigate's stats endpoint shows detector inference speed:

```bash
curl -s http://localhost:5001/api/stats | jq '.detectors.cpu.inference_speed'
```

Target: **< 50ms** per inference. If it's consistently > 80ms, you need to reduce resolution, FPS, or camera count.

### 9.5 Memory Pressure Check

```bash
# Check SHM usage:
df -h /dev/shm

# Check Frigate process memory:
ps -eo pid,comm,rss --sort=-rss | head -20
# RSS is in KB — Frigate total (all processes) should be < 2 GB on this system

# Check for swap usage (bad — indicates RAM pressure):
free -h | grep Swap
# If Swap "used" is growing, reduce camera count or SHM frame count
```

---

## Quick-Start Checklist

- [ ] Install `mesa-va-drivers` and verify with `vainfo`
- [ ] Set `shm_size: "256mb"` in docker-compose
- [ ] Use `type: cpu` detector with `num_threads: 3`
- [ ] Set detect resolution to 640×360 and FPS to 5
- [ ] Use `preset-vaapi` for `hwaccel_args`
- [ ] Disable birdseye, audio detection, semantic search, face recognition, LPR
- [ ] Limit ffmpeg to 2 threads (`global_args: [-threads, "2"]`)
- [ ] Set recording retention to 3–7 days to manage SSD space
- [ ] Add motion masks for areas with constant false motion (trees, flags)
- [ ] Set `vm.swappiness=10` on the host
- [ ] Monitor inference speed via `/api/stats` — target < 50ms
- [ ] Monitor CPU temp — if > 85°C sustained, scale back

---

## References

- [Frigate Hardware Recommendations](https://docs.frigate.video/guides/getting_started#hardware)
- [VAAPI on AMD GPUs](https://wiki.archlinux.org/title/Hardware_video_acceleration#AMD)
- [Picasso APU VCN Capabilities](https://en.wikipedia.org/wiki/Video_Coding_Engine)
- [TensorFlow Lite XNNPACK delegate](https://www.tensorflow.org/lite/performance/xnnpack_delegate)
