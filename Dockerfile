# ---- Stage 0: micromamba shim ----
FROM mambaorg/micromamba:1.5.6 AS micromamba

# ---- Stage 1: build a MinkowskiEngine wheel (Torch 2.0.1 + cu118) ----
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS build

ARG ME_JOBS=4
ENV MAX_JOBS=${ME_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${ME_JOBS} \
    MAKEFLAGS=-j${ME_JOBS}

WORKDIR /build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 build-essential cmake curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# micromamba bootstrap
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV PATH=/opt/conda/bin:$PATH
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"

# base env (no torch yet)
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 mkl numpy omegaconf torchmetrics laspy lazrs-python \
    && micromamba clean --all --yes

# Torch 2.0.1 + cu118 and PyG wheels compatible with it
RUN micromamba run -n base python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN micromamba run -n base pip install \
    torch-geometric==2.3.1 torch-scatter==2.1.1 torch-sparse==0.6.17 \
    torch-cluster==1.6.1 torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Build ME with CUDA & OpenBLAS
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA=1
ENV CC=gcc
ENV CXX=g++

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ninja-build libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    micromamba run -n base python -c "import torch, sys, os; \
    print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Py:', sys.version.split()[0]); \
    print('torch.cuda.is_available:', torch.cuda.is_available(), 'CUDA_HOME:', os.environ.get('CUDA_HOME'))" && \
    # MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    micromamba run -n base python -m pip wheel -v --no-build-isolation --no-cache-dir \
    --config-settings=--build-option=--force_cuda \
    --config-settings=--build-option=--blas=openblas \
    --config-settings=--build-option=--blas_include_dirs=/usr/include/x86_64-linux-gnu \
    . && \
    mv *.whl /build/

# ---- Stage 2: slim runtime with prebuilt ME wheel ----
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 AS runtime

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV PATH=/opt/conda/bin:$PATH
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"

# Ensure OpenBLAS is present at runtime for the ME wheel
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# copy the MinkowskiEngine wheel we just built
COPY --from=build /build/*.whl /wheels/

WORKDIR /app

# minimal base libs via conda, then pip the GPU stack + ME wheel
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 mkl numpy omegaconf torchmetrics laspy lazrs-python && \
    micromamba clean --all --yes && \
    micromamba run -n base python -m pip install -U pip wheel ninja && \
    micromamba run -n base python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 && \
    micromamba run -n base pip install \
    torch-geometric==2.3.1 torch-scatter==2.1.1 torch-sparse==0.6.17 \
    torch-cluster==1.6.1 torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html && \
    ls -lh /wheels && \
    micromamba run -n base python -m pip install --no-cache-dir --no-index \
    --find-links=/wheels MinkowskiEngine && \
    micromamba run -n base python - <<'PY'
import torch, MinkowskiEngine as ME
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda,
      "| CUDA avail:", torch.cuda.is_available(), "| ME import OK")
PY

# copy only experiment0 code
COPY experiment0/*.py /app/
COPY experiment0/configs /app/configs
COPY experiment0/scripts /app/scripts

ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash"]
