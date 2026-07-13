# libncnn with Vulkan + C API, for direct Rust FFI inference (no Python
# subprocess). SIMPLEVK bundles ncnn's own Vulkan loader, so no libvulkan-dev
# is needed at build time; at runtime it dlopens the container's libvulkan.
FROM debian:bookworm-slim AS ncnn-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Tag must match the pip ncnn version (1.0.YYYYMMDD) the models are
# validated against — older libncnn mis-executes newer YOLO exports.
RUN git clone --depth 1 --branch 20260526 https://github.com/Tencent/ncnn.git /ncnn \
    && cd /ncnn && git submodule update --init --recursive --depth 1
RUN cmake -S /ncnn -B /ncnn/build -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_SHARED_LIB=ON -DNCNN_C_API=ON \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF \
    && cmake --build /ncnn/build -j"$(nproc)" \
    && cmake --install /ncnn/build --prefix /opt/ncnn-install \
    && cp -L /opt/ncnn-install/lib/libncnn.so /libncnn.so

FROM rust:slim-bookworm AS rust-builder
WORKDIR /build

# Copy and compile Rust components
COPY frigate-detector-rs/ frigate-detector-rs/
COPY frigate-motion-rs/ frigate-motion-rs/
COPY frigate-yolo-rs/ frigate-yolo-rs/
COPY frigate-frame-rs/ frigate-frame-rs/

RUN cd frigate-detector-rs && cargo build --release
RUN cd frigate-motion-rs && cargo build --release
RUN cd frigate-yolo-rs && cargo build --release
RUN cd frigate-frame-rs && cargo build --release

# Web build stage
FROM --platform=$BUILDPLATFORM node:20-slim AS web-builder
WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN NODE_OPTIONS="--max-old-space-size=4096" npm run build

# Final stage
FROM ghcr.io/blakeblackshear/frigate:stable
RUN pip install --break-system-packages ncnn

# Copy Rust detector and libraries from builder
COPY --from=ncnn-builder /libncnn.so /opt/frigate/libncnn.so
COPY --from=rust-builder /build/frigate-detector-rs/target/release/frigate-detector-rs /opt/frigate/frigate-detector-rs
COPY --from=rust-builder /build/frigate-motion-rs/target/release/libfrigate_motion_rs.so /opt/frigate/libfrigate_motion_rs.so
COPY --from=rust-builder /build/frigate-yolo-rs/target/release/libfrigate_yolo_rs.so /opt/frigate/libfrigate_yolo_rs.so
COPY --from=rust-builder /build/frigate-frame-rs/target/release/libfrigate_frame_rs.so /opt/frigate/libfrigate_frame_rs.so

# Copy migrations
COPY migrations/ /opt/frigate/migrations/

# Copy the entire frigate python codebase
COPY frigate/ /opt/frigate/frigate/

# Copy custom ncnn-vulkan plugin
COPY docker/ncnn-vulkan/onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py

# Copy prebuilt web assets
COPY --from=web-builder /work/dist/ /opt/frigate/web/

RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
