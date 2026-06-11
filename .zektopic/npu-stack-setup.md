# NPU Driver Stack Setup — Ryzen 8600G on Linux

> **Target hardware:** AMD Ryzen 5 8600G (Phoenix, XDNA 1, PCI 1022:1502)
> **OS:** Linux Mint / Ubuntu with kernel ≥ 6.14
> **Goal:** Get `xrt-smi examine` showing "NPU Phoenix" and run inference

---

## 0. Prerequisites

```bash
# Verify kernel version (need 6.14+)
uname -r                    # Should be ≥ 6.14.0

# Verify NPU appears on PCI bus
lspci | grep -i IPU         # Should show "AMD IPU Device" at 10:00.1
lspci -nn | grep 1022:1502  # Alternative: check by PCI ID

# Verify /dev/accel/accel0 exists
ls -la /dev/accel/accel0    # crw-rw---- 1 root render 261, 0

# Check dmesg for driver
dmesg | grep amdxdna        # Should show driver loaded, firmware loaded

# Check firmware
ls /lib/firmware/amdnpu/    # Should contain npu.sbin files
```

If `/dev/accel/accel0` doesn't exist:
```bash
# Check kernel config
grep AMDXDNA /boot/config-$(uname -r)    # Should be y or m
# If =m (module): modprobe amdxdna

# Check dmesg for errors
dmesg | grep -i "amdxdna\|npu\|ipu" | tail -30
```

---

## 1. Build and Install XRT (Xilinx Runtime)

XRT provides the userspace runtime for XDNA NPU devices.

```bash
# Clone XRT (use master for latest fixes)
git clone https://github.com/Xilinx/XRT.git
cd XRT
git submodule update --init --recursive

# Create build directory and build
mkdir -p build/Release
cd build/Release

# Configure - only build NPU support (skip Alveo FPGA stuff)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DXRT_NPU=1 \
      -DCMAKE_INSTALL_PREFIX=/opt/xilinx/xrt \
      ../../src

# Build with all cores
make -j$(nproc)

# Install (still in build/Release)
sudo make install

# Or package and install
make package
# This produces .tar.gz files (NOT .deb in newer XRT versions)
# -- xrt_<version>-base.tar.gz (the main package)
# -- xrt_<version>-npu.tar.gz  (NPU-specific files)

# Manual install from tar:
sudo mkdir -p /opt/xilinx/xrt
sudo tar -xzf xrt_*base.tar.gz -C /
sudo tar -xzf xrt_*npu.tar.gz -C /

# Set up environment
source /opt/xilinx/xrt/setup.sh
# Add to ~/.bashrc for persistence:
echo 'source /opt/xilinx/xrt/setup.sh' >> ~/.bashrc
```

**Troubleshooting:**
- "XILINX_VITIS is undefined" warning → safe to ignore with `-noert` flag
- No .deb produced → normal in newer XRT, use tar.gz
- Build fails with missing dependencies → run `sudo ./tools/amdxdna_deps.sh` from xdna-driver repo

---

## 2. Build and Install xdna-driver Plugin

The plugin bridges XRT to the amdxdna kernel driver.

```bash
# Clone xdna-driver
git clone https://github.com/amd/xdna-driver.git
cd xdna-driver
git submodule update --init --recursive

# Install build dependencies
sudo ./tools/amdxdna_deps.sh

# Build the plugin
cd build
./build.sh -release

# This should produce:
# xrt_plugin.<version>-x86_64-amdxdna.deb (or .tar.gz)
# Install it:
sudo apt install ./xrt_plugin.*-amdxdna.deb   # If .deb
# OR if .tar.gz:
sudo tar -xzf xrt_plugin.*-amdxdna.tar.gz -C /
```

**Troubleshooting:**
- If plugin build fails with "xrt-base" dependency error → XRT not installed properly, re-source setup.sh
- If `.deb` install fails on dependency → use `dpkg --force-depends -i` or extract tar manually

---

## 3. Verify the Stack

```bash
# Source XRT environment (if not already done)
source /opt/xilinx/xrt/setup.sh

# Fix permissions on /dev/accel/accel0 if needed
sudo chmod 666 /dev/accel/accel0
# Or add your user to render group:
sudo usermod -a -G render $USER
# (log out and back in after this)

# Test: Enumerate devices
xrt-smi examine
# Expected output:
#   Device(s) Present
#   |BDF             |Name         |
#   |[0000:10:00.1]  |NPU Phoenix  |

# Test: Run validation
xrt-smi validate
# Should run test workloads and report PASSED

# Check detailed info
xrt-smi examine -d 0000:10:00.1
# Shows firmware version, XRT version, NPU capabilities
```

**If "0 devices found" (like issue #686):**
```bash
# 1. Check kernel driver is bound
ls -la /sys/bus/pci/devices/0000:10:00.1/driver

# 2. Check IOMMU is enabled
dmesg | grep -i iommu
# Should show "AMD-Vi: Interrupt remapping enabled"

# 3. Try newer firmware
# Check amd/xdna-driver releases page for latest npu.sbin
sudo cp npu.sbin /lib/firmware/amdnpu/1502_00/
sudo rmmod amdxdna && sudo modprobe amdxdna

# 4. Check for driver binding issues
sudo lspci -s 10:00.1 -vvv | grep -i driver

# 5. If still no device, it may be a known issue with XDNA 1
#    Check: https://github.com/amd/xdna-driver/issues
#    The GPU path (ROCm on Radeon 760M) still works
```

---

## 4. Build ONNX Runtime with Vitis AI EP

This is the inference engine that Frigate will use.

```bash
# Clone ONNX Runtime
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Vitis AI EP requires specific build flags
./build.sh \
    --config Release \
    --use_vitisai \
    --build_shared_lib \
    --parallel $(nproc)

# The build produces libonnxruntime.so with Vitis AI EP baked in
# Install to system:
sudo cp build/Linux/Release/libonnxruntime*.so* /usr/local/lib/
sudo ldconfig

# Verify Vitis AI EP is available:
python3 -c "
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'VitisAIExecutionProvider'
"
```

**Alternative: Pre-built wheels (if available)**
```bash
# AMD may provide pre-built ONNX Runtime wheels with Vitis AI EP
pip install onnxruntime-vitisai
# Check if this exists: https://pypi.org/search/?q=onnxruntime+vitisai
```

---

## 5. Compile YOLOv8n Model for NPU

```bash
# The model needs to be compiled for the NPU's AIE architecture
# This generates an xclbin + compiled model

# Option A: Use Vitis AI Model Zoo (if YOLOv8 is available)
# https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo

# Option B: Use the Vitis AI compiler
# This requires the full Vitis AI toolkit
python3 -m vai_q_onnx.quantize \
    --input_model yolov8n_320x320.onnx \
    --output_model yolov8n_320x320_quantized.onnx \
    --calib_dataset /path/to/calibration/images

# Compile for NPU
python3 -m vai_c_onnx \
    --model yolov8n_320x320_quantized.onnx \
    --arch /opt/xilinx/xrt/arch/arch.json \
    --output_dir compiled_model/ \
    --net_name yolov8n_npu

# This produces:
# compiled_model/yolov8n_npu.xclbin     — Hardware bitstream
# compiled_model/yolov8n_npu.onnx       — Compiled model
```

---

## 6. Standalone Inference Test

```python
"""Test NPU inference outside of Frigate."""
import numpy as np
import onnxruntime as ort

# Configure Vitis AI EP
providers = [("VitisAIExecutionProvider", {
    "device": "NPU",
    "config_file": "/path/to/vaip_config.json",
    "cache_dir": "/tmp/npu_cache",
})]

session = ort.InferenceSession(
    "compiled_model/yolov8n_npu.onnx",
    providers=providers,
)

# Run inference
dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: dummy_input})

print(f"Input shape: {dummy_input.shape}")
print(f"Output shapes: {[o.shape for o in outputs]}")
print("NPU inference successful!")
```

---

## 7. Persistent Environment Setup

Add to `/etc/environment` or Frigate's docker-compose:

```yaml
environment:
  - XILINX_XRT=/opt/xilinx/xrt
  - LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/local/lib
  - PATH=/opt/xilinx/xrt/bin:$PATH
  - XRT_HACK_UNSECURE_LOADING_XCLBIN=1  # May be needed for dev
```

For Docker, bind-mount the XRT and NPU device:
```yaml
volumes:
  - /opt/xilinx/xrt:/opt/xilinx/xrt:ro
  - /dev/accel/accel0:/dev/accel/accel0
  - /dev/dri:/dev/dri  # Still needed for VAAPI decode
devices:
  - /dev/accel:/dev/accel
```

---

## 8. Known Issues & Workarounds

| Issue | Symptom | Status |
|-------|---------|--------|
| `xrt-smi` shows 0 devices | Driver loaded but XRT can't talk to NPU | [Issue #686](https://github.com/amd/xdna-driver/issues/686) — unresolved as of Aug 2025 |
| XRT builds produce .tar.gz not .deb | `apt install` fails with "can only specify package archive" | Normal in newer XRT, use `dpkg -i` or extract manually |
| Vitis AI EP not in ONNX Runtime | Provider list doesn't include VitisAI | Ensure `--use_vitisai` flag during ORT build |
| NPU firmware load fails | `dmesg` shows firmware errors | Try different firmware version from xdna-driver releases |
| Permission denied on /dev/accel/accel0 | Can't access NPU as non-root | `usermod -a -G render $USER` or `chmod 666` |

---

## References

- [[npu-migration-strategy]] — Overall migration plan
- [[npu-detector-design]] — Frigate detector plugin design
- [amd/xdna-driver README](https://github.com/amd/xdna-driver)
- [Xilinx/XRT](https://github.com/Xilinx/XRT)
- [ONNX Runtime Vitis AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
