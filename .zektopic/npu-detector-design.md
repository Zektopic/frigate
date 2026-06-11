# NPU Detector Plugin Design — Frigate

> **File:** `frigate/detectors/plugins/xdnapu.py`
> **Detector type key:** `xdnapu`
> **Config key in config.yml:** `detectors: {name}: {type: xdnapu, ...}`

---

## 1. Architecture Overview

The XDNA NPU detector follows the same pattern as all Frigate detectors — subclass `DetectionApi`, implement `detect_raw()`, register via auto-discovery.

```
config.yml → FrigateConfig → DetectorConfig (type: xdnapu)
    → create_detector() → XDNANPUDetector.__init__()
        → ONNX Runtime session with Vitis AI EP
        → Model compiled for XDNA NPU
    → detect_raw(tensor) → preprocess → NPU inference → postprocess
        → np.ndarray(20, 6) [class_id, score, y1, x1, y2, x2]
```

### Inference Stack
```
Frigate (Python)
    ↓
ONNX Runtime (C++)
    ↓  Vitis AI Execution Provider
XRT (C++/Python)
    ↓  xdna-driver plugin
amdxdna kernel driver (/dev/accel/accel0)
    ↓
XDNA NPU hardware
```

---

## 2. Config Schema

```yaml
detectors:
  npu:
    type: xdnapu
    device: NPU                    # NPU or CPU (fallback)
    model_path: /config/model_cache/yolov8/yolov8n_320x320.onnx
    # NPU-specific options (optional)
    xclbin: auto                   # Path to XCLBIN, or "auto" to use default
    compile_only: false            # If true, compile model but don't run
    cache_dir: /config/model_cache/xdnapu  # Compiled model cache
```

### Pydantic Model

```python
class XDNANPUDetectorConfig(BaseDetectorConfig):
    type: Literal["xdnapu"]
    device: str = Field(default="NPU", title="Device (NPU or CPU)")
    xclbin: str = Field(default="auto", title="Path to XCLBIN file")
    compile_only: bool = Field(default=False, title="Compile model only")
    cache_dir: str = Field(default="/config/model_cache/xdnapu",
                            title="Compiled model cache directory")
```

---

## 3. Detector Class Design

```python
class XDNANPUDetector(DetectionApi):
    type_key = "xdnapu"
    supported_models = [ModelTypeEnum.yolo_generic, ModelTypeEnum.yolox]

    def __init__(self, detector_config: XDNANPUDetectorConfig):
        # 1. Set up XRT environment
        # 2. Discover NPU device
        # 3. Create ONNX Runtime session with Vitis AI EP
        # 4. Load and compile model
        # 5. Warmup inference

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # 1. Preprocess (NHWC→NCHW if needed, normalize)
        # 2. Run ONNX Runtime inference on NPU
        # 3. Post-process (filter, NMS, format)
        # Returns: np.ndarray shape (20, 6)
```

---

## 4. Key Design Decisions

### 4.1 ONNX Runtime vs IREE
**Choice: ONNX Runtime with Vitis AI EP**

- Frigate already supports ONNX models natively
- Vitis AI EP is AMD's supported path for NPU inference
- IREE+MLIR-AIE is more active open-source but requires model re-compilation to MLIR
- ONNX Runtime path allows fallback to CPU if NPU unavailable

### 4.2 Model Compilation Strategy
- **First run:** Load ONNX → compile for XDNA → cache compiled model to disk
- **Subsequent runs:** Load cached compiled model
- **Cache key:** Model hash + XRT version + xclbin version
- **Separate ONNX files per model** (already the Frigate convention)

### 4.3 Error Handling
- If NPU device not found → log warning, fall back to CPU inference
- If model compilation fails → log error, raise so watchdog can restart
- If inference times out (>5s) → log error, return empty detections
- NPU driver crash → watchdog detects hung process, restarts detector

### 4.4 Memory Management
- NPU uses system RAM via IOMMU — no dedicated VRAM
- Model weights: ~3-5 MB (YOLOv8n)
- Input tensor: 640×360×3 = ~700KB
- Output tensor: 20×6 float32 = 480 bytes
- Total per-inference footprint: < 50 MB

### 4.5 Concurrency
- XDNA NPU is single-queue, sequential execution
- Frigate's multiprocessing model already sends frames one at a time via `detection_queue`
- No need for batching or concurrent inference
- If multiple detectors share one NPU → XRT handles queuing

---

## 5. Implementation Steps

### Step 1: Environment Detection (lines ~30-60)
```python
def _setup_xrt_environment(self):
    """Set up XRT and locate NPU device."""
    import pyxrt  # XRT Python bindings
    xrt_source = "/opt/xilinx/xrt/setup.sh"
    # XRT must be sourced before import
    # Check for NPU device
    devices = pyxrt.device.enumerate()
    if not devices:
        raise RuntimeError("No XDNA NPU device found")
    self.device = devices[0]
    logger.info("XDNA NPU device found: %s", self.device.get_info())
```

### Step 2: ONNX Runtime Session (lines ~62-100)
```python
def _create_session(self, model_path: str):
    """Create ONNX Runtime session with Vitis AI EP."""
    import onnxruntime as ort

    providers = [
        ("VitisAIExecutionProvider", {
            "device": self.config.device,
            "config_file": self._get_vaip_config(),
            "cache_dir": self.config.cache_dir,
        }),
        "CPUExecutionProvider",  # Fallback
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    self.session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )
```

### Step 3: Preprocessing (lines ~102-130)
```python
def _preprocess(self, tensor_input: np.ndarray) -> np.ndarray:
    """Preprocess input tensor for NPU inference."""
    # Ensure 4D: (1, H, W, 3) for NHWC, or (1, 3, H, W) for NCHW
    if tensor_input.ndim == 3:
        tensor_input = np.expand_dims(tensor_input, axis=0)

    if self.input_tensor == "nchw":
        tensor_input = np.transpose(tensor_input, (0, 3, 1, 2))

    # Normalize if needed (float models expect 0-1 range)
    if self.input_dtype == "float":
        tensor_input = tensor_input.astype(np.float32) / 255.0

    return tensor_input
```

### Step 4: Postprocessing (reuse existing)
```python
def _postprocess(self, raw_output: np.ndarray) -> np.ndarray:
    """Post-process NPU output to Frigate format (20, 6)."""
    # Reuse existing YOLO post-processing from frigate/util/model.py
    # Same as ONNX detector: filter by confidence, NMS, normalize coords
    ...
```

### Step 5: detect_raw (lines ~150-180)
```python
def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
    """Run object detection on the NPU."""
    preprocessed = self._preprocess(tensor_input)

    input_name = self.session.get_inputs()[0].name
    output_names = [o.name for o in self.session.get_outputs()]

    raw_output = self.session.run(output_names, {input_name: preprocessed})

    detections = self._postprocess(raw_output)
    return detections
```

---

## 6. Testing Strategy

### Unit Tests (`frigate/test/test_detector_xdnapu.py`)
- Test config validation (valid/invalid xclbin paths)
- Test preprocessing (NHWC→NCHW transpose, normalization)
- Test postprocessing with mock NPU output
- Test error handling (no device, compilation failure)
- Test empty frame handling

### Integration Tests
- Mock XRT device (or test on actual hardware)
- Verify ONNX model loads and compiles
- Verify inference produces expected output shape
- Compare detection results against CPU detector baseline

### Benchmarks
- Inference latency (target: <50ms per frame)
- Memory usage (target: <50MB per detector process)
- 6-camera sustained throughput at 5fps

---

## 7. Open Questions

1. **Vitis AI EP availability:** Does ONNX Runtime Vitis AI EP work with XDNA 1 (Phoenix) NPU, or only XDNA 2 (Strix Point)?
   - *Fallback:* IREE with AMD-AIE plugin

2. **Model format:** Does the NPU require model quantization (INT8) or can it run FP16/FP32?
   - *Likely:* INT8 quantization required for optimal performance

3. **XCLBIN compilation:** Can we use a pre-compiled xclbin or must we compile from source for each model?
   - *Likely:* Pre-compiled xclbin for YOLO-type models exists in Vitis AI model zoo

4. **Multi-process safety:** Can two detector processes share the NPU?
   - *Likely no:* Single context per NPU. If needed, use a single NPU detector process with higher throughput.

5. **Power states:** Does the NPU have low-power states that add latency on first inference after idle?
   - *Unknown:* Test on actual hardware

---

## 8. References

- [ONNX Runtime Vitis AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [Vitis AI ONNX Runtime Examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/onnxrt)
- [Frigate Detector API](frigate/detectors/detection_api.py)
- [MIGraphX Detector Reference](frigate/detectors/plugins/migraphx.py)
- [ONNX Detector Reference](frigate/detectors/plugins/onnx.py)
- [[npu-migration-strategy]] — Overall migration plan
- [[npu-stack-setup]] — NPU driver stack setup guide
