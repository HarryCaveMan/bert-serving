FROM amazonlinux:2023 as cuda_plugins

RUN dnf -y update && dnf -y upgrade && dnf -y install python3.11 python3.11-pip \
    &&\
    pip3.11 install \
    nvidia-nccl-cu12 \
    nvidia-cudnn-cu12 \
    tensorrt

FROM amazonlinux:2023 as cuda_base

ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_HOME=$CUDA_ROOT
ENV CUDA_INCLUDE_PATH=$CUDA_ROOT/include
ENV CUDA_LIBRARY_PATH=$CUDA_ROOT/lib64
ENV CUDA_BIN=$CUDA_ROOT/bin
ENV TENSORRT_ROOT=/usr/local/tensorrt
ENV TENSORRT_LIB_PATH=$TENSORRT_ROOT/lib
ENV TENSORRT_INCLUDE_PATH=$TENSORRT_ROOT/include
ENV TENSORRT_BIN_PATH=$TENSORRT_ROOT/bin
ENV PATH=$CUDA_BIN:$TENSORRT_BIN_PATH:$PATH
ENV LD_LIBRARY_PATH=$CUDA_LIBRARY_PATH:$TENSORRT_LIB_PATH:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_INCLUDE_PATH:$TENSORRT_INCLUDE_PATH:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_INCLUDE_PATH:$TENSORRT_INCLUDE_PATH:$CPLUS_INCLUDE_PATH

RUN dnf -y update && dnf -y upgrade &&\
    dnf -y install 'dnf-command(config-manager)' &&\
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo &&\
    dnf -y update && dnf -y install cuda-toolkit &&\
    dnf -y install python3.11 python3.11-setuptools python3.11-devel &&\
    ln /usr/bin/python3.11 /usr/bin/python

COPY --from=cuda_plugins /usr/local/lib/python3.11/site-packages/nvidia/nccl/lib/* $CUDA_LIBRARY_PATH
COPY --from=cuda_plugins /usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib/* $CUDA_LIBRARY_PATH
COPY --from=cuda_plugins /usr/local/lib/python3.11/site-packages/nvidia/nccl/include/* $CUDA_INCLUDE_PATH
COPY --from=cuda_plugins /usr/local/lib/python3.11/site-packages/nvidia/cudnn/include/* $CUDA_INCLUDE_PATH
COPY libs/TensorRT-10.7.0.23/ $TENSORRT_ROOT/

FROM cuda_base as build

ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /opt
RUN dnf -y update && dnf -y upgrade &&\
    dnf -y install \
      openssl-devel \
      clang \
      pkg-config \
      git \
      git-lfs \
      unzip \
      python3.11 \
      python3.11-pip \
    &&\
    pip3.11 install \
      tensorrt \
      onnx \
      transformers \
      diffusers \
      accelerate \
      pillow \
      jupyterlab \
      scikit-learn \
      matplotlib \
    &&\
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs| bash -s -- -y &&\
    rustup component add rust-src &&\
    cargo install --locked evcxr_jupyter &&\
    evcxr_jupyter --install

WORKDIR /opt/workspace

CMD ["jupyter","lab","--NotebookApp.token=''","--NotebookApp.password=''","--allow-root","--ip","0.0.0.0"]