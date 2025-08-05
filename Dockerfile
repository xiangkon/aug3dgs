# -------------------- 基础镜像 --------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 安装所有系统依赖（合并为一个RUN指令以减少镜像层）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # 编译工具
        wget git build-essential ca-certificates \
        ninja-build g++ cmake \
        # OpenGL/Vulkan 相关
        colmap libx11-6 libgl1-mesa-glx libglew2.2 \
        libglu1-mesa mesa-utils xvfb libvulkan1 \
        libgl1 libglx-mesa0 \
        # 多媒体/图形
        ffmpeg vulkan-tools \
        # Qt/Boost 开发库
        qtbase5-dev libboost-all-dev \
        # GLib 开发
        libglib2.0-0 libglib2.0-dev && \
    # 清理缓存减小镜像体积
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
# 安装 Miniconda（官方脚本）
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 在创建环境前添加
RUN conda config --system --remove channels defaults && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict

# 创建 conda 环境并激活
RUN conda create -n tdgs_aug python=3.10 -y
SHELL ["conda", "run", "-n", "tdgs_aug", "/bin/bash", "-c"]

# 安装 PyTorch（CUDA 12.1）
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 克隆仓库
WORKDIR /workspace
RUN git clone https://github.com/xiangkon/aug3dgs.git && \
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git

# 安装 aug3dgs 依赖
WORKDIR /workspace/aug3dgs
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


# 设置 CUDA 编译环境
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 安装子模块
RUN pip install submodule/diff-gaussian-rasterization

# 安装 pytorch3d（官方 conda 包）
RUN wget -q https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2 && \
    conda install -y ./pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2 && \
    rm pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2

# 安装 gaussian-splatting 子模块
WORKDIR /workspace/gaussian-splatting
RUN pip install submodules/simple-knn submodules/fused-ssim

# 设置默认工作目录和 shell
WORKDIR /workspace/aug3dgs
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate tdgs_aug" >> ~/.bashrc
ENV BASH_ENV=~/.bashrc

# 默认命令
CMD ["/bin/bash"]