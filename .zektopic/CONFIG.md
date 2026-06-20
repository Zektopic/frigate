# Frigate Configuration Reference

Working production config for AMD Radeon 760M with ncnn Vulkan GPU detection.

## Full config.yml

```yaml
mqtt:
  enabled: false

go2rtc:
  streams:
    front_camera:
      - rtsp://root:PASSWORD@CAMERA_IP/stream=0
    # ... add your cameras here

ffmpeg:
  hwaccel_args: preset-vaapi
  # VAAPI is SAFE with Vulkan — only ROCm caused the deadlock

detectors:
  ncnn_vulkan:
    type: onnx
    device: GPU

model:
  model_type: yolo-generic
  width: 640
  height: 640
  # Must match ncnn model input size (yolov5s = 640x640)
  input_tensor: nchw
  input_dtype: float
  path: /config/model_cache/yolov9-s-320.onnx
  labelmap_path: /labelmap/coco-80.txt

record:
  enabled: true
  continuous:
    days: 3
  motion:
    days: 3
  alerts:
    retain:
      days: 3
      mode: all
  detections:
    retain:
      days: 3
      mode: all

cameras:
  front_camera:
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:8554/front_camera
          input_args: preset-rtsp-restream
          roles:
            - detect
            - record
    detect:
      enabled: true
      width: 1920
      height: 1080
      fps: 10
  # ... repeat for each camera

semantic_search:
  enabled: true
  model_size: small

face_recognition:
  enabled: true
  model_size: small

lpr:
  enabled: true
  model_size: small

classification:
  bird:
    enabled: true
```

## docker-compose.yml

```yaml
services:
  frigate:
    container_name: frigate
    restart: unless-stopped
    image: frigate-ncnn-vulkan
    shm_size: "1gb"
    network_mode: host
    volumes:
      - ./config:/config
      - ./storage:/media/frigate
      - /etc/localtime:/etc/localtime:ro
      - type: tmpfs
        target: /tmp/cache
        tmpfs:
          size: 1000000000
    environment:
      FRIGATE_RTSP_PASSWORD: ${FRIGATE_RTSP_PASSWORD:-}
      OMP_NUM_THREADS: "2"
      ORT_NUM_THREADS: "2"
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    # NOTE: /dev/kfd is NOT needed — Vulkan doesn't use ROCm
```

## Key Config Decisions

### Why `type: onnx`?

The ncnn plugin hijacks the `onnx` detector type to avoid needing to modify Frigate's config validation schema. The detector key (`ncnn_vulkan`) is just a label.

### Why `width: 640, height: 640`?

Must match the ncnn model's expected input size. YOLOv5s uses 640×640. A mismatch causes incorrect detections.

### Why `fps: 10`?

The GPU can handle ~30 inferences/second at 125ms each. With 4 cameras at 10 FPS = 40 inferences/sec, there's some frame skipping but it's reasonable.

### Why VAAPI is safe now?

The GPU hang was caused by ROCm's kernel driver, not VAAPI. With Vulkan (RADV), VAAPI media decode and Vulkan compute coexist without issues — verified for 3+ hours.

### Why OMP/ORT threads at 2?

These affect ONNX Runtime CPU operations (face recognition, semantic search embeddings), not ncnn GPU inference. Limiting prevents those models from saturating CPU.

## Model Cache Files

```
config/model_cache/
├── yolov5s.ncnn.param    # ncnn model (YOLOv5s)
├── yolov5s.ncnn.bin      # ncnn weights
├── yolov9-s-320.onnx     # kept for Frigate config validation
├── arcface.onnx          # face recognition
├── facedet.onnx          # face detection
└── jinaai/               # semantic search (CLIP)
```
