FROM ghcr.io/blakeblackshear/frigate:stable
RUN pip install --break-system-packages ncnn
COPY onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py
RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
