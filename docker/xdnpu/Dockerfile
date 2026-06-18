# syntax=docker/dockerfile:1.4

# AMD XDNA NPU + ROCm combined image for Phoenix1 APUs
# Builds on the ROCm image and adds NPU-specific OpenVINO configuration

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM=6.4.3

#######################################################################
FROM wget AS rocm

ARG ROCM

RUN apt update -qq && \
    apt install -y wget gpg && \
    wget -O rocm.deb https://repo.radeon.com/amdgpu-install/${ROCM}/ubuntu/jammy/amdgpu-install_6.4.60403-1_all.deb && \
    apt install -y ./rocm.deb && \
    apt update && \
    apt install -qq -y rocm

RUN mkdir -p /opt/rocm-dist/opt/rocm-${ROCM}/lib
RUN cd /opt/rocm-${ROCM}/lib && \
    cp -dpr libMIOpen*.so* libamd*.so* libhip*.so* libhsa*.so* libmigraphx*.so* librocm*.so* librocblas*.so* libroctracer*.so* librocsolver*.so* librocfft*.so* librocprofiler*.so* libroctx*.so* /opt/rocm-dist/opt/rocm-${ROCM}/lib/ && \
    mkdir -p /opt/rocm-dist/opt/rocm-${ROCM}/lib/migraphx/lib && \
    cp -dpr migraphx/lib/* /opt/rocm-dist/opt/rocm-${ROCM}/lib/migraphx/lib 2>/dev/null || true
RUN cd /opt/rocm-dist/opt/ && ln -s rocm-${ROCM} rocm

RUN mkdir -p /opt/rocm-dist/etc/ld.so.conf.d/
RUN echo /opt/rocm/lib|tee /opt/rocm-dist/etc/ld.so.conf.d/rocm.conf

#######################################################################
FROM deps AS deps-prelim

COPY docker/rocm/debian-backports.sources /etc/apt/sources.list.d/debian-backports.sources
RUN echo "deb http://deb.debian.org/debian trixie main" > /etc/apt/sources.list.d/trixie.list && \
    apt-get update && \
    apt-get install -y libnuma1 && \
    apt-get install -qq -y -t bookworm-backports mesa-va-drivers mesa-vulkan-drivers && \
    apt-get install -qq -y -t trixie libstdc++-14-dev && \
    rm -f /etc/apt/sources.list.d/trixie.list && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/frigate
COPY --from=rootfs / /

RUN wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && sed -i 's/args.append("setuptools")/args.append("setuptools==77.0.3")/' get-pip.py \
    && python3 get-pip.py "pip" --break-system-packages
RUN python3 -m pip config set global.break-system-packages true

# Install ROCm ONNX Runtime
COPY docker/rocm/requirements-wheels-rocm.txt /requirements-rocm.txt
RUN pip3 uninstall -y onnxruntime \
    && pip3 install -r /requirements-rocm.txt

#######################################################################
FROM scratch AS rocm-dist

ARG ROCM

COPY --from=rocm /opt/rocm-${ROCM}/include/ /opt/rocm-${ROCM}/include/
COPY --from=rocm /opt/rocm-$ROCM/bin/rocminfo /opt/rocm-$ROCM/bin/migraphx-driver /opt/rocm-$ROCM/bin/
COPY --from=rocm /opt/rocm-$ROCM/share/miopen/db/*gfx10* /opt/rocm-$ROCM/share/miopen/db/
COPY --from=rocm /opt/rocm-$ROCM/share/miopen/db/*gfx11* /opt/rocm-$ROCM/share/miopen/db/
COPY --from=rocm /opt/rocm-$ROCM/share/miopen/db/*gfx12* /opt/rocm-$ROCM/share/miopen/db/
COPY --from=rocm /opt/rocm-$ROCM/share/miopen/db/*gfx90* /opt/rocm-$ROCM/share/miopen/db/
COPY --from=rocm /opt/rocm-$ROCM/lib/rocblas/library/*gfx10* /opt/rocm-$ROCM/lib/rocblas/library/
COPY --from=rocm /opt/rocm-$ROCM/lib/rocblas/library/*gfx11* /opt/rocm-$ROCM/lib/rocblas/library/
COPY --from=rocm /opt/rocm-$ROCM/lib/rocblas/library/*gfx12* /opt/rocm-$ROCM/lib/rocblas/library/
COPY --from=rocm /opt/rocm-$ROCM/lib/rocblas/library/*gfx90* /opt/rocm-$ROCM/lib/rocblas/library/
COPY --from=rocm /opt/rocm-dist/ /

#######################################################################
FROM deps-prelim AS final

ENV MIGRAPHX_DISABLE_MIOPEN_FUSION=1
ENV MIGRAPHX_DISABLE_SCHEDULE_PASS=1
ENV MIGRAPHX_DISABLE_REDUCE_FUSION=1
ENV MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS=1
ENV MIOPEN_CUSTOM_CACHE_DIR=/config/model_cache/migraphx
ENV MIOPEN_USER_DB_PATH=/config/model_cache/migraphx
ENV AMD_COMGR_CACHE=1
ENV AMD_COMGR_CACHE_DIR=/config/model_cache/migraphx

COPY --from=rocm-dist / /

RUN ldconfig

# NPU device access note:
# The AMD XDNA NPU is accessed via /dev/accel/accel0
# Ensure the container has this device mounted and the render group has access.
