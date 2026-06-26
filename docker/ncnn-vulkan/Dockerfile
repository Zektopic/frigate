FROM ghcr.io/blakeblackshear/frigate:stable

# Install ncnn with Vulkan support
RUN pip install --break-system-packages ncnn

# Copy ncnn Vulkan detector (replaces ONNX detector)
COPY onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py

# ── Rust SIMD-accelerated motion detection engine ──────────────────
COPY libfrigate_motion_rs.so /opt/frigate/
COPY libfrigate_frame_rs.so /opt/frigate/
COPY rust_engine.py /opt/frigate/frigate/motion/rust_engine.py

# ── Phase 1: CPU/RAM-optimized Python modules ──────────────────────
COPY ffmpeg.py /opt/frigate/frigate/video/ffmpeg.py
COPY sqlitevecq.py /opt/frigate/frigate/db/sqlitevecq.py
COPY app.py /opt/frigate/frigate/app.py
COPY auth.py /opt/frigate/frigate/api/auth.py
COPY services.py /opt/frigate/frigate/util/services.py
COPY storage.py /opt/frigate/frigate/storage.py
COPY ws.py /opt/frigate/frigate/comms/ws.py
COPY ffmpeg_presets.py /opt/frigate/frigate/ffmpeg_presets.py
COPY base.py /opt/frigate/frigate/object_detection/base.py

# Clear Python cache so our replacements take effect
RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
