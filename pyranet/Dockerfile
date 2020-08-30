FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y --allow-change-held-packages \
            libnccl2=2.5.6-1+cuda9.0 \
            libnccl-dev=2.5.6-1+cuda9.0 && \
    apt-get install -y \ 
            wget \
            git \
            sudo \ 
            software-properties-common \
            libssl-dev \
            libzmq3-dev \
            python-dev \
            python-pip \
            gcc \
            libhdf5-serial-dev \
            libmatio-dev \
            graphviz

# Install nccl v1 for cuda 8.0
RUN cd /root && \
            wget https://github.com/NVIDIA/nccl/releases/download/v1.3.4-1/libnccl1_1.3.4-1.cuda8.0_amd64.deb && \
            dpkg -i libnccl1_1.3.4-1.cuda8.0_amd64.deb && \
            apt-get install -f
RUN cd /root && \
            wget https://github.com/NVIDIA/nccl/releases/download/v1.3.4-1/libnccl-dev_1.3.4-1.cuda8.0_amd64.deb && \
            dpkg -i libnccl-dev_1.3.4-1.cuda8.0_amd64.deb && \
            apt-get install -f

# Install Jupyter Notebook for iTorch
RUN pip install --upgrade pip
RUN pip install jupyter

# Run Torch installation scripts
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && \
    cd /root/torch && \
    bash install-deps && \
    ./install.sh -b

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install lua packages
RUN luarocks install itorch
RUN luarocks install hdf5
RUN luarocks install matio
RUN luarocks install optnet
RUN luarocks install nccl

# ENV CUDNN_PATH="/usr/lib/x86_64-linux-gnu/libcudnn.so.7"
RUN git clone https://github.com/soumith/cudnn.torch.git -b R7 /root/cudnn.torch && \
    cd /root/cudnn.torch && \
    luarocks make cudnn-scm-1.rockspec

WORKDIR /app

