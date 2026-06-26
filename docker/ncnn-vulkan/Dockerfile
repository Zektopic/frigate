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

# Final stage
FROM ghcr.io/blakeblackshear/frigate:stable
RUN pip install --break-system-packages ncnn

# Copy Rust detector and libraries from builder
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

RUN find /opt/frigate -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/frigate -name "*.pyc" -delete 2>/dev/null
