# RustingStrata server image for NVIDIA (CUDA) and Vulkan GPUs. docs/HOSTING.md, "Docker", has the run commands.
#   docker build -t rustingstrata .
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
# .cargo/config.toml builds for the local CPU; an image runs on other machines too, so it targets x86-64-v3
# (AVX2 and FMA: Intel Haswell, AMD Zen and later), like the release binaries.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_DEBUG=0
RUN cargo build --release --locked

FROM nvidia/cuda:12.9.1-base-ubuntu24.04
# NVRTC compiles the CUDA kernels at the first start (12.x also serves sm_70 cards, which 13.x dropped). The Vulkan
# loader and the Mesa drivers are for --backend vulkan; the NVIDIA Vulkan driver comes from the host and needs
# libXext and libEGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvrtc-12-9 libvulkan1 mesa-vulkan-drivers libxext6 libegl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/rustingstrata /usr/local/bin/
# The runtime package has no unversioned libnvrtc.so, so the engine is given the name. /cache holds the calibration,
# the compiled kernels and the expert placement: mount it, or every new container starts without them.
ENV STRATA_NVRTC=libnvrtc.so.12 XDG_CACHE_HOME=/cache NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
VOLUME ["/cache", "/models"]
EXPOSE 8080
ENTRYPOINT ["rustingstrata"]
CMD ["serve", "--gpu", "--addr", "0.0.0.0:8080", "--models-dir", "/models"]
