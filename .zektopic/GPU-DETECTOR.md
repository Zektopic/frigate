# NCNN Vulkan GPU Detector

Drop-in replacement for Frigate's ONNX detector using ncnn with Vulkan backend via RADV.

## Architecture

```
Camera frame → Frigate preprocessing → ncnn Vulkan → RADV → GPU
                                                         ↑
                                              (no ROCm involved)
```

Unlike ROCm/MIGraphX, this uses the Mesa RADV Vulkan driver — the same driver every Linux game uses. RADV is production-grade and has working GPU reset.

## Setup

### 1. Build the custom image

```bash
cd docker/ncnn-vulkan
docker build -t frigate-ncnn-vulkan .
```

### 2. Download the ncnn YOLO model

```bash
cd config/model_cache/
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.param
curl -LO https://github.com/nihui/ncnn-assets/raw/master/models/yolov5s.ncnn.bin
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
  path: /config/model_cache/yolov9-s-320.onnx  # keeps Frigate validation happy
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

1. **Plugin discovery**: Frigate scans `detectors/plugins/` for Python modules. Our `onnx_ncnn_vulkan.py` registers with `type_key = "onnx"`, hijacking the ONNX detector slot.

2. **Model loading**: ncnn loads the YOLOv5s model (.param + .bin) at startup. GPU is selected automatically via Vulkan device enumeration.

3. **Inference**: Each frame arrives preprocessed from Frigate (resized to 640×640, RGB, NCHW format, normalized to [0,1]). We multiply by 255 to convert back to [0,255] (ncnn YOLOv5s expects raw pixel values). ncnn runs the model on GPU via Vulkan compute shaders.

4. **Postprocessing**: We apply sigmoid to the raw logit outputs, then use Frigate's own `post_process_yolo()` function for grid decoding and NMS — identical to the ONNX path.

## Critical Bug — Normalization

**The ncnn YOLOv5s model expects input in [0, 255] range!** Frigate normalizes frames to [0, 1] by dividing by 255. If you forget to multiply back by 255, the model produces near-zero confidence scores and no detections.

```python
# Fix in detect_raw():
mat_in = self.ncnn.Mat((tensor_input.squeeze(0) * 255.0).astype(np.float32))
```

## Performance (Radeon 760M / Phoenix1)

| Resolution | Inference | GPU usage | Detection rate |
|---|---|---|---|
| 640×640 | ~125ms | ~40% | 7-8 FPS (4 cameras) |
| 320×320 | ~35ms | ~7% | 2-5 FPS (4 cameras) |

The 640×640 model is more accurate but slower. The 320×320 model is faster but was designed for YOLOv9-s, not YOLOv5s.

## Verified Hardware

| GPU | Driver | ROCm? | Status |
|---|---|---|---|
| AMD Radeon 760M (gfx1103) | RADV GFX1103_R1 | Broken | ✅ Stable |
| Any Vulkan 1.2+ GPU | Mesa RADV | N/A | ✅ Expected |
| Vega 8 (gfx909) | RADV | None | ✅ Expected (~80-120ms) |

## Model Compatibility

The plugin hijacks the ONNX detector type, so any model path in the Frigate config passes validation. The actual model files loaded by ncnn are the `.param` and `.bin` files in the same directory. Currently hardcoded to `yolov5s.ncnn.{param,bin}`.

To use a different model (e.g., YOLOv8n, YOLO11n):
1. Download the `.param` and `.bin` from [ncnn-assets](https://github.com/nihui/ncnn-assets/tree/master/models)
2. Update the filenames in the plugin's `__init__` method
3. The output layer names might differ — check with `net.output_names()`
