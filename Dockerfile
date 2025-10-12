# Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# micromamba bootstrap
ARG MAMBA_ROOT_PREFIX=/opt/micromamba
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:$PATH
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/mm.tar.bz2 && \
    mkdir -p ${MAMBA_ROOT_PREFIX} && \
    tar -xvjf /tmp/mm.tar.bz2 -C /usr/local/bin bin/micromamba --strip-components=1

# Create/populate the base env (note: install, not create)
RUN micromamba install -y -n base -c pytorch -c nvidia -c conda-forge \
    python=3.10 pytorch=2.3.1 pytorch-cuda=12.1 mkl numpy omegaconf torchmetrics laspy lazrs-python && \
    micromamba clean -a -y

# 2) Install PyTorch 2.3.1 + cu121 via pip (reliable)
RUN micromamba run -n base python -m pip install -U pip wheel ninja && \
    micromamba run -n base python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \
    micromamba run -n base python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

# PyG wheels matching torch 2.3.1 + cu121
RUN micromamba run -n base pip install -U pip wheel && \
    micromamba run -n base pip install \
    torch-geometric==2.5.3 torch-scatter==2.1.2 torch-sparse==0.6.18 \
    torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# Build MinkowskiEngine for Ada (RTX 4060 = sm_89)
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    MAX_JOBS=$(nproc) micromamba run -n base python -m pip install -v --no-cache-dir \
    --config-settings=--build-option=--force_cuda .

WORKDIR /app
COPY experiment0/*.py /app/
COPY experiment0/configs /app/configs
COPY experiment0/scripts /app/scripts

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
