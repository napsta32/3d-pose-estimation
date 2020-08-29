FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y \ 
            git \
            sudo \ 
            software-properties-common \
            libssl-dev \
            libzmq3-dev \
            python-dev \
            python-pip \
            gcc \
            libhdf5-serial-dev

# Install Jupyter Notebook for iTorch
RUN pip install --upgrade pip
RUN pip install jupyter

# Run Torch installation scripts
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && \
    cd /root/torch && \
    bash install-deps
RUN cd /root/torch && \
    TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" ./install.sh -b

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

# Install nccl for better performance
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make && \
    make install && \
    luarocks install nccl

# Set ~/torch as working directory
WORKDIR /root/torch