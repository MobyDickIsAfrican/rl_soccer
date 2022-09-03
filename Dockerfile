# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1
FROM python:3.8

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN pip install -U 'mujoco-py<2.2,>=2.1'
RUN mkdir -p /root/.mujoco
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz 
RUN tar -xvzf mujoco.tar.gz -C /root/.mujoco 
RUN rm mujoco.tar.gz
COPY mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

WORKDIR /dm_soccer2gym
COPY dm_soccer2gym/ /dm_soccer2gym
RUN python /dm_soccer2gym/setup.py install


WORKDIR /soccer_proyect
COPY soccer_proyect/ /soccer_proyect
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends libopenmpi-dev
RUN pip install -r requirements.txt
RUN pip install protobuf==3.20.*
RUN pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html