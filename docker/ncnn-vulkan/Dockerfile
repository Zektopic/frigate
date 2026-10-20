# Multi-arch: the ncnn and Rust stages run on the build host and
# cross-compile for TARGETARCH (amd64 or arm64), so an arm64 image needs QEMU
# only for the final pip install, not for the long compile steps.

# libncnn with Vulkan + C API, for direct Rust FFI inference (no Python
# subprocess). SIMPLEVK bundles ncnn's own Vulkan loader, so no libvulkan-dev
# is needed at build time; at runtime it dlopens the container's libvulkan.
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS ncnn-builder
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake g++ make ca-certificates \
    && if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get install -y --no-install-recommends g++-aarch64-linux-gnu; \
    fi \
    && rm -rf /var/lib/apt/lists/*
# Tag must match the pip ncnn version (1.0.YYYYMMDD) the models are
# validated against — older libncnn mis-executes newer YOLO exports.
RUN git clone --depth 1 --branch 20260526 https://github.com/Tencent/ncnn.git /ncnn \
    && cd /ncnn && git submodule update --init --recursive --depth 1
# arm64 uses ncnn's own cross toolchain file; ncnn picks NEON/fp16/dotprod
# kernels at runtime from the CPU it runs on (Pi 4 A72 through Pi 5 A76).
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=/ncnn/toolchains/aarch64-linux-gnu.toolchain.cmake"; \
    fi \
    && cmake -S /ncnn -B /ncnn/build -DCMAKE_BUILD_TYPE=Release $TOOLCHAIN \
        -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_SHARED_LIB=ON -DNCNN_C_API=ON \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF \
    && cmake --build /ncnn/build -j"$(nproc)" \
    && cmake --install /ncnn/build --prefix /opt/ncnn-install \
    && cp -L /opt/ncnn-install/lib/libncnn.so /libncnn.so

FROM --platform=$BUILDPLATFORM rust:slim-bookworm AS rust-builder
ARG TARGETARCH
WORKDIR /build
RUN case "$TARGETARCH" in \
        amd64) echo x86_64-unknown-linux-gnu > /rust-target ;; \
        arm64) echo aarch64-unknown-linux-gnu > /rust-target \
            && apt-get update \
            && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
            && rm -rf /var/lib/apt/lists/* ;; \
        *) echo "unsupported TARGETARCH $TARGETARCH" && exit 1 ;; \
    esac \
    && rustup target add "$(cat /rust-target)"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

# Copy and compile Rust components
COPY frigate-detector-rs/ frigate-detector-rs/
COPY frigate-motion-rs/ frigate-motion-rs/
COPY frigate-yolo-rs/ frigate-yolo-rs/
COPY frigate-frame-rs/ frigate-frame-rs/

# AVX2 paths in frigate-motion-rs are selected at runtime and compiled out
# on aarch64, so the same sources build for both targets.
RUN T="$(cat /rust-target)" && mkdir -p /build/out \
    && for crate in detector motion yolo frame; do \
        (cd frigate-$crate-rs && cargo build --release --target "$T") || exit 1; \
    done \
    && cp frigate-detector-rs/target/$T/release/frigate-detector-rs /build/out/ \
    && for crate in motion yolo frame; do \
        cp frigate-$crate-rs/target/$T/release/libfrigate_${crate}_rs.so /build/out/; \
    done

# Web build stage
FROM --platform=$BUILDPLATFORM node:20-slim AS web-builder
WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN NODE_OPTIONS="--max-old-space-size=4096" npm run build

# Final stage
FROM ghcr.io/blakeblackshear/frigate:stable
RUN pip install --break-system-packages --no-deps ncnn portalocker tqdm

# Copy Rust detector and libraries from builder
COPY --from=ncnn-builder /libncnn.so /opt/frigate/libncnn.so
COPY --from=rust-builder /build/out/frigate-detector-rs /opt/frigate/frigate-detector-rs
COPY --from=rust-builder /build/out/libfrigate_motion_rs.so /opt/frigate/libfrigate_motion_rs.so
COPY --from=rust-builder /build/out/libfrigate_yolo_rs.so /opt/frigate/libfrigate_yolo_rs.so
COPY --from=rust-builder /build/out/libfrigate_frame_rs.so /opt/frigate/libfrigate_frame_rs.so

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
