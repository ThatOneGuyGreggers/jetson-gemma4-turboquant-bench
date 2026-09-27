FROM docker.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 AS build

ARG TURBOQUANT_COMMIT=a3d5603d110bda29222d2011596cdc84d7fa532d
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake g++-14 gcc-14 git libgomp1 libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
WORKDIR /src
RUN git init \
    && git remote add origin https://github.com/TheTom/llama-cpp-turboquant.git \
    && git fetch --depth 1 origin "$TURBOQUANT_COMMIT" \
    && git checkout --detach FETCH_HEAD

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_USE_PREBUILT_UI=ON \
    && cmake --build build --config Release --target llama-server -j6 \
    && mkdir -p /out/bin /out/lib \
    && cp build/bin/llama-server /out/bin/ \
    && find build -name "*.so*" -exec cp -P {} /out/lib/ \;

FROM docker.io/nvidia/cuda:13.2.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /usr/local/cuda/compat_orin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /app/
ENV LD_LIBRARY_PATH=/app/lib
WORKDIR /app
ENTRYPOINT ["/bin/bash", "-c", "cp -P /app/lib/*.so* /app/bin/ && exec /app/bin/llama-server \"$@\"", "--"]
