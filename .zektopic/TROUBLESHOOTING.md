# Troubleshooting

Common issues and fixes for the ncnn Vulkan detector.

## No detections (events not triggering)

### Symptom
Detection FPS shows in stats but `detected_objects` is always null.

### Fix
Check the model input size matches the config:
```yaml
model:
  width: 640   # must match ncnn model
  height: 640
```

Verify the normalization fix is present in the detector:
```python
# Must multiply by 255 — ncnn YOLOv5s expects 0-255 range
mat_in = self.ncnn.Mat((tensor_input.squeeze(0) * 255.0).astype(np.float32))
```

### Debug
```bash
docker exec frigate python3 -c "
import cv2, numpy as np, ncnn
# Test with bus.jpg
img = cv2.imread('/tmp/bus.jpg')
img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
tensor = np.expand_dims(img.transpose(2,0,1).astype(np.float32), 0)
net = ncnn.Net(); net.load_param('/config/model_cache/yolov5s.ncnn.param'); net.load_model('/config/model_cache/yolov5s.ncnn.bin')
mat = ncnn.Mat(tensor.squeeze(0))
with net.create_extractor() as ex:
    ex.input('in0', mat)
    _, out0 = ex.extract('out0')
    _, out1 = ex.extract('out1')
    _, out2 = ex.extract('out2')
outputs = [np.expand_dims(1/(1+np.exp(-np.array(m))), 0) for m in [out0, out1, out2]]
from frigate.util.model import post_process_yolo
r = post_process_yolo(outputs, 640, 640)
print(f'Detections: {np.count_nonzero(r[:, 1])}')  # Should be 15-20 for bus.jpg
"
```

## GPU not detected by ncnn

### Symptom
```
NCNN: no GPU found, using CPU fallback
```

### Fix
1. Verify Mesa Vulkan drivers are installed:
   ```bash
   apt install mesa-vulkan-drivers
   ```
2. Verify ncnn can see the GPU:
   ```bash
   docker exec frigate python3 -c "import ncnn; print(ncnn.get_gpu_count())"
   ```
3. If count is 0, check `/dev/dri/renderD128` is passed through in docker-compose

## Container won't start (NaN in output)

### Symptom
```
ValueError: could not broadcast input array from shape (0,6) into shape (20,6)
```

### Fix
The model is producing zero valid detections. This is almost always the normalization issue — check the `* 255.0` fix in the detector code. Test with bus.jpg as above to isolate.

## Address already in use (OSError 98)

### Symptom
```
OSError: [Errno 98] Address already in use
```

### Explanation
This is Frigate's internal auth service port conflict during startup. It's harmless — Frigate retries and starts correctly. Not related to the ncnn detector.

## Detection is slow (>200ms)

### Symptom
Inference speed >200ms per frame.

### Fix
1. Verify Vulkan is enabled: `docker logs frigate | grep "Vulkan=on"`
2. If using CPU fallback, check `/dev/dri/renderD128` passthrough
3. Reduce model input size (but requires re-converting the model)
4. Check GPU isn't thermal throttling: `rocm-smi` or `sensors`

## Face recognition crashes (segfault)

### Symptom
```
SIGSEGV in ArcFace model
```

### Fix
This only happened with MIGraphX/ROCm EP. With ncnn Vulkan, object detection uses GPU and face recognition uses ONNX Runtime CPU EP — no segfaults. If it occurs unexpectedly, add the model type to `is_migraphx_complex_model` exclusion list.

## After reboot, everything works but no recent events

### Normal
Events are only created when objects are detected. If no one has walked by a camera since reboot, there won't be events. Test by walking in front of a camera.

### Verify detection is working
```bash
curl -s http://localhost:5000/api/stats | jq '.detectors."ncnn_vulkan".inference_speed'
# Should show actual value (not 10.0 placeholder)
```
