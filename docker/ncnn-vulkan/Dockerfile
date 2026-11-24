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
COPY frigate-stats-rs/ frigate-stats-rs/

# AVX2 paths in frigate-motion-rs are selected at runtime and compiled out
# on aarch64, so the same sources build for both targets.
RUN T="$(cat /rust-target)" && mkdir -p /build/out \
    && for crate in detector motion yolo frame stats; do \
        (cd frigate-$crate-rs && cargo build --release --target "$T") || exit 1; \
    done \
    && cp frigate-detector-rs/target/$T/release/frigate-detector-rs /build/out/ \
    && for crate in motion yolo frame stats; do \
        cp frigate-$crate-rs/target/$T/release/libfrigate_${crate}_rs.so /build/out/; \
    done

# Web build stage
FROM --platform=$BUILDPLATFORM node:20-slim AS web-builder
WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN NODE_OPTIONS="--max-old-space-size=4096" npm run build

# The base image ships go2rtc built with go1.25.6 and golang.org/x/crypto
# v0.47.0, tempio from 2021 (go1.17.1) and yq built with go1.25.4, which
# together carry 9 critical Go stdlib/x-module CVEs. Rebuild pinned upstream
# tags with the current Go release instead. The go2rtc release binary is
# UPX-packed, so building from source also keeps it visible to image scanners.
FROM --platform=$BUILDPLATFORM golang:1.27.4-bookworm AS go-tools-builder
ARG TARGETARCH
ARG GO2RTC_VERSION=v1.9.15
ARG TEMPIO_VERSION=2026.07.0
ARG YQ_VERSION=v4.54.3
ENV CGO_ENABLED=0 GOOS=linux GOFLAGS=-trimpath
RUN git clone --depth 1 --branch "$GO2RTC_VERSION" https://github.com/AlexxIT/go2rtc /src/go2rtc \
    && cd /src/go2rtc && GOARCH="$TARGETARCH" go build -ldflags "-s -w" -o /out/go2rtc .
# tempio 2026.07.0 still pins golang.org/x/crypto v0.31.0 (via sprig)
RUN git clone --depth 1 --branch "$TEMPIO_VERSION" https://github.com/home-assistant/tempio /src/tempio \
    && cd /src/tempio && go get golang.org/x/crypto@v0.59.0 && go mod tidy \
    && GOARCH="$TARGETARCH" go build -ldflags "-s -w" -o /out/tempio .
RUN git clone --depth 1 --branch "$YQ_VERSION" https://github.com/mikefarah/yq /src/yq \
    && cd /src/yq && GOARCH="$TARGETARCH" go build -ldflags "-s -w" -o /out/yq .

# Final stage
FROM ghcr.io/blakeblackshear/frigate:stable
# Pick up Debian bookworm security updates published after the base image
# was built (curl, OpenSSL, krb5, glib, libxml2, ...). Plain upgrade only:
# no new packages and no removals.
RUN apt-get update \
    && apt-get -y --no-install-recommends upgrade \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=go-tools-builder /out/go2rtc /usr/local/go2rtc/bin/go2rtc
COPY --from=go-tools-builder /out/tempio /usr/local/tempio/bin/tempio
COPY --from=go-tools-builder /out/yq /usr/local/bin/yq
# Pinned: --no-deps skips resolver checks, so an unpinned upgrade could
# silently pull an incompatible or compromised release into the image
RUN pip install --break-system-packages --no-deps \
    ncnn==1.0.20260526 portalocker==4.4.1 tqdm==4.70.1

# Copy Rust detector and libraries from builder
COPY --from=ncnn-builder /libncnn.so /opt/frigate/libncnn.so
COPY --from=rust-builder /build/out/frigate-detector-rs /opt/frigate/frigate-detector-rs
COPY --from=rust-builder /build/out/libfrigate_motion_rs.so /opt/frigate/libfrigate_motion_rs.so
COPY --from=rust-builder /build/out/libfrigate_yolo_rs.so /opt/frigate/libfrigate_yolo_rs.so
COPY --from=rust-builder /build/out/libfrigate_frame_rs.so /opt/frigate/libfrigate_frame_rs.so
COPY --from=rust-builder /build/out/libfrigate_stats_rs.so /opt/frigate/libfrigate_stats_rs.so

# Copy migrations
COPY migrations/ /opt/frigate/migrations/

# Copy the entire frigate python codebase
COPY frigate/ /opt/frigate/frigate/

# The base image's go2rtc config generator tracks upstream's Python: since
# 0.18.3 it imports helpers (is_valid_ffmpeg_path, apply_go2rtc_ffmpeg_config)
# that this fork's frigate/ does not have, so go2rtc never got a config. Ship
# the fork's own copy, which is kept in step with frigate/ above.
COPY docker/main/rootfs/usr/local/go2rtc/create_config.py /usr/local/go2rtc/create_config.py

# Copy custom ncnn-vulkan plugin
COPY docker/ncnn-vulkan/onnx.py /opt/frigate/frigate/detectors/plugins/onnx.py

# Copy prebuilt web assets
COPY --from=web-builder /work/dist/ /opt/frigate/web/

# The base image's nginx config sets $go2rtc_stream_proxy only inside the go2rtc
# stream locations, but the /auth subrequest of every request reads it. nginx
# then logs "using uninitialized variable" once per request (about two thirds
# of all log lines on a 9-camera system). Give it an empty default at server
# level; the stream locations still override it with 1, and an empty value
# sends no header, exactly as before. Skipped if the base image stops using
# the variable or already initializes it.
RUN conf=/usr/local/nginx/conf/auth_location.conf; \
    if grep -q 'go2rtc_stream_proxy' "$conf" \
        && ! grep -q '^set \$go2rtc_stream_proxy' "$conf"; then \
        sed -i '1i set $go2rtc_stream_proxy "";' "$conf"; \
    fi

RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
