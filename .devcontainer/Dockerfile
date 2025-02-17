# Run the following command to build `flemington7/self-recognition:latest`
# docker build -f .devcontainer/Dockerfile -t flemington7/self-recognition:latest .
ARG CUDA=12.1.0
ARG DIST=ubuntu20.04
ARG TARGET=cudnn8-devel
FROM --platform=linux/amd64 nvidia/cuda:${CUDA}-${TARGET}-${DIST}

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="Asia/Shanghai"

# Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    language-pack-en \
    make \
    man-db \
    manpages \
    manpages-dev \
    manpages-posix \
    manpages-posix-dev \
    sudo \
    unzip \
    vim \
    wget \
    fish \
    parallel \
    iputils-ping \
    htop \
    emacs \
    zsh \
    rsync \
    tmux

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Make the base image friendlier for interactive workloads. This makes things like the man command work.
RUN yes | unminimize

RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh

# Initialize conda so that the shell (e.g. when running bash -l) will have conda activated.
RUN /bin/bash -c "source /root/miniconda3/bin/activate && conda init --all"

# Add conda to the PATH
ENV PATH="/root/miniconda3/bin:/root/miniconda3/condabin:${PATH}"

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel 
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging
RUN pip install flash-attn==2.7.2.post1 --no-build-isolation
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

COPY open_instruct open_instruct

# install the package in editable mode
COPY pyproject.toml .
RUN pip install -e .
COPY .git/ ./.git/
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /workspace

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]