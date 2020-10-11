FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y \ 
            wget \
            git \
            sudo \
            gcc \
            make \
            graphviz

# Install miniconda 3
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes
ENV PATH /opt/conda/bin:$PATH

# Install conda dependencies
RUN conda install numpy matplotlib scipy Cython
RUN conda install -c anaconda pytorch-gpu
RUN conda install -c pytorch torchvision
RUN conda install -c conda-forge opencv
RUN conda install -c anaconda h5py

# Set coco workdir
WORKDIR /root/app
ENV COCO_HOME="/root/coco"
ENV PYTHONPATH=$PYTHONPATH:$COCO_HOME

# Install COCOAPI
RUN git clone https://github.com/cocodataset/cocoapi.git $COCO_HOME/lib/COCOAPI \
    && cd $COCO_HOME/lib/COCOAPI/PythonAPI \
    && make install

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
ENV MSPN_HOME="/app"
ENV PYTHONPATH=$PYTHONPATH:$MSPN_HOME
