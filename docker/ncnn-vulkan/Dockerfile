FROM ghcr.io/blakeblackshear/frigate:stable

# Install ncnn with Vulkan support
RUN pip install --break-system-packages ncnn

# Copy ncnn Vulkan detector (replaces ONNX detector)
COPY onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py

# Clear Python cache so our replacement takes effect
RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
