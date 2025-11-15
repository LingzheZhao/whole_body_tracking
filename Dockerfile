FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG MAX_JOBS 32
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=120;100;90;89;86;80
ENV TORCH_CUDA_ARCH_LIST="12.0;10.0;9.0;8.9;8.6;8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_CACHE_DIR=/tmp/
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    MUJOCO_GL=osmesa
# 3.10 for isaac-sim 4.x and 3.11 for isaac-sim 5.x
ARG PYTHON_VERSION=3.11
RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined requirements and common tools / libs
        autoconf automake autotools-dev build-essential ca-certificates gnupg2 \
        make cmake yasm pkg-config gdb gcovr binutils-dev libiberty-dev \
        ninja-build ccache cppcheck doxygen graphviz plantuml \
        cimg-dev \
        clang-format \
        dh-autoreconf \
        libatlas-base-dev \
        libboost-all-dev \
        liblog4cplus-dev \
        libssh2-1-dev \
        libreadline-dev \
        libcereal-dev \
        libcgal-dev \
        libblas-dev libcxsparse3 libceres-dev libeigen3-dev libfftw3-dev liblapack-dev liblapacke-dev libnlopt-dev \
        libflann-dev metis libmetis-dev libopenblas-dev libsuitesparse-dev \
        libgtk2.0-dev libglew-dev libglm-dev libglu1-mesa-dev mesa-utils \
        freeglut3-dev libassimp-dev libglfw3-dev libproj-dev libqt5opengl5-dev \
        libxrandr-dev libxxf86vm-dev \
        libgflags-dev libgoogle-glog-dev libgtest-dev \
        libfreeimage-dev libjpeg-dev libpng-dev libtiff-dev libtiff5-dev \
        libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libv4l-dev v4l-utils \
        libpcap-dev libpostproc-dev libpq-dev libswscale-dev libxine2-dev \
        intel-mkl libopenmpi-dev libtbb2 libtbb-dev  \
        daemontools krb5-user libkrb5-dev \
        ibverbs-providers libibverbs1 libnuma1 libnuma-dev libpmi2-0-dev \
        libhdf5-dev protobuf-compiler libprotobuf-dev librdmacm1 libsqlite3-dev libssl-dev libtool \
        libyaml-dev libyaml-cpp-dev libicu-dev libsimpleini-dev \
        libpcl-dev \
        libopencv-dev libopenexr-dev \
        ffmpeg zlib1g-dev \
        ## Useful tools
        git curl wget unzip xz-utils nano vim-tiny \
        sudo htop iftop iotop \
        net-tools netcat iputils-ping dnsutils nmap \
        cloc rsync screen tmux \
        daemontools nfs-common rsync proxychains \
        openssh-server openssh-client \
        software-properties-common \
        # Python
        python-is-python3 \
        python${PYTHON_VERSION}-dev python3-pip \
        # python3-numpy \
        # VTK
        libvtk9-dev \
        # OSMesa build dependencies
        libosmesa6-dev \
        # EGL build dependencies
        libgl1-mesa-glx \
        libopengl-dev \
        libglvnd-dev \
        libglib2.0-0 \
        libgl-dev \
        libglx-dev \
        libegl-dev \
        # X11 utils
        mesa-utils \
        x11-apps \
        xorg-dev \
        # QT
        qtbase5-dev \
        # additional Vulkan
        libatomic1 \
        libegl1 \
        libglu1-mesa \
        libgomp1 \
        libsm6 \
        libxi6 \
        libxrandr2 \
        libxt6 \
        libfreetype-dev \
        libfontconfig1 \
        openssl \
    # && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
    && apt-get purge unattended-upgrades \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Determined AI stuff
#! ---EDIT notebook-requirements.txt TO ADD PYPI PACKAGES----
WORKDIR /tmp
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.2.3 &&\
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    pip install -r additional-requirements.txt &&\
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements" &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

ARG VULKAN_SDK_VERSION=1.3.296.0
# Add LunarG Vulkan repository and install Vulkan SDK
ARG DEBIAN_FRONTEND=noninteractive
RUN curl -sSL https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /usr/share/keyrings/lunarg-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/lunarg-archive-keyring.gpg] https://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION%.*}/ jammy main" > /etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_SDK_VERSION%.*}-jammy.list && \
    apt update && \
    apt install -y vulkan-sdk && \
    apt-get -y autoremove && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Setup the required capabilities for the container runtime    
# ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Open ports for live streaming
EXPOSE 47998/udp \
       49100/tcp \
       47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       49100/tcp \
       8011/tcp \
       8012/tcp \
       8211/tcp \
       8899/tcp \
       8891/tcp

WORKDIR /opt

# Default entrypoint to launch headless with streaming
# ENTRYPOINT /isaac-sim/runheadless.native.sh

# Install ROS2
# Which ROS2 apt package to install
ARG ROS2_APT_PACKAGE=desktop
# ROS2 Humble Apt installations
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo jammy) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-${ROS2_APT_PACKAGE} \
    ros-humble-vision-msgs \
    # Install both FastRTPS and CycloneDDS
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rmw-fastrtps-cpp \
    # This includes various dev tools including colcon
    ros-dev-tools &&\
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    # Add sourcing of setup.bash to .bashrc
    echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-gl-570 \
    && apt-get -y autoremove && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN chmod 777 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/omni &&\
    mkdir /.nvidia-omniverse &&\
    chmod 777 /.nvidia-omniverse

# Install Unitree SDK
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd cyclonedds && cmake -GNinja -B build && \
    cmake --build build -t install && \
    cd .. && rm -rf cyclonedds

RUN git clone --recursive https://github.com/unitreerobotics/unitree_sdk2/ &&\
    cd unitree_sdk2 &&\
    cmake -GNinja -B build &&\
    cmake --build build --target install &&\
    cd .. && rm -rf unitree_sdk2
ENV CYCLONEDDS_HOME=/usr/local
RUN git clone --recursive https://github.com/unitreerobotics/unitree_sdk2_python/ &&\
    cd unitree_sdk2_python &&\
    pip install -e .

# Install Isaac Sim
RUN pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com &&\
    pip install "torch==2.7.0+cu128" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Fix matplotlib installation to use apt version to avoid conflicts
RUN apt-get remove -y python3-matplotlib && \
    apt-get clean autoclean && rm -rf /var/lib/apt/lists/* && \
    pip install "numpy==1.26.1" matplotlib --force-reinstall && \
    pip install git+https://github.com/stack-of-tasks/pinocchio.git && \
    pip install pin-pink==3.1.0 pin==2.7.0
RUN pip install "rerun-sdk==0.23.1" "numpy==1.26.1" zmq flatdict &&\
    pip install logging_mp onnxruntime toml ruamel-yaml pathspec determined==0.32.0 &&\
    echo "alias omni_python='/isaac-sim/python.sh'" >> ~/.bashrc &&\
    # Fix Multiple Installable Client Drivers (ICDs) issue
    rm /usr/share/vulkan/icd.d/nvidia_icd.json

# Install Isaac Lab
ARG ISAACLAB_PATH=/opt/IsaacLab
ARG TERM="xterm"
RUN git clone https://github.com/isaac-sim/IsaacLab.git
# COPY scripts/disable_rsl_rl_dep.patch /tmp/disable_rsl_rl_dep.patch
# RUN cd IsaacLab && \
#     git config --global --add safe.directory /opt/IsaacLab && \
#     git checkout v1.4.1 && \
#     git apply /tmp/disable_rsl_rl_dep.patch && \
#     TERM="xterm" bash ./isaaclab.sh --install

COPY scripts/do_not_check_pytorch.patch /tmp/do_not_check_pytorch.patch
RUN cd IsaacLab && \
    git config --global --add safe.directory /opt/IsaacLab && \
    git checkout v2.2.0 && \
    git apply /tmp/do_not_check_pytorch.patch && \
    TERM="xterm" bash ./isaaclab.sh -i &&\
    # Install rosdeps for extensions that declare a ros_ws in
    # their extension.toml
    ./isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py rosdep ${ISAACLAB_PATH}/source

RUN git clone https://github.com/leggedrobotics/rsl_rl.git && \
    cd rsl_rl && \
    git checkout v2.0.1 && \
    pip install -e .
# RUN git clone https://github.com/zitongbai/rsl_rl.git && \
#     cd rsl_rl && \
#     git checkout feature/amp && \
#     pip install -e . &&\
#     cd .. &&\
#     git clone --recursive https://github.com/zitongbai/legged_lab && \
#     cd legged_lab && \
#     pip install -e source/legged_lab
COPY ./legged_lab/data/ /opt/legged_lab/source/legged_lab/legged_lab/data/

# Set environment variables for Isaac Sim and Isaac Lab
ENV PYTHONPATH=${PYTHON_PATH}:/opt/IsaacLab/source/isaaclab:/opt/IsaacLab/source/isaaclab_assets/:/opt/IsaacLab/source/isaaclab_mimic/:/opt/IsaacLab/source/isaaclab_rl/:/opt/IsaacLab/source/isaaclab_tasks/:/isaac-sim/python_packages/ \
    omni_python='/isaac-sim/python.sh'