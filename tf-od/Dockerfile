FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

WORKDIR /root

# Compile protobuf configs
RUN mkdir -p /root/models/research/
COPY object_detection /root/models/research/object_detection
RUN (cd /root/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /root/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/root/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

WORKDIR /root
RUN python -m pip install tf_slim

# CMD [ "jupyter", "notebook", "--allow-root", "--no-browser", "--port", "8090", "--ip", "0.0.0.0" ]

WORKDIR /root/project
CMD [ "python", "-u", "extract-bb.py" ]