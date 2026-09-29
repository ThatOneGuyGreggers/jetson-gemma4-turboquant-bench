FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake gcc-14 g++-14 git libssl-dev && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
ARG LLAMA_COMMIT=fc07d781e61f0d23764394e902b88d26a974e202
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /src && \
    cd /src && git checkout "$LLAMA_COMMIT"
WORKDIR /src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=ON && \
    cmake --build build --target llama-server -j4 && \
    mkdir -p /out/bin /out/lib && cp build/bin/llama-server /out/bin/ && \
    find build -name '*.so*' -exec cp -P {} /out/lib/ \;

FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /usr/local/cuda/compat_orin /var/lib/apt/lists/*
COPY --from=build /out/ /app/
ENV LD_LIBRARY_PATH=/app/lib
ENTRYPOINT ["/bin/bash", "-c", "cp -P /app/lib/*.so* /app/bin/ && exec /app/bin/llama-server \"$@\"", "--"]
