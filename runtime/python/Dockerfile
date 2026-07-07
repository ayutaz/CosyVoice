FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++
RUN git lfs install
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
# NOTE dependencies are managed in pyproject.toml/uv.lock, export them for the image python
RUN cd CosyVoice && pip3 install uv --no-cache-dir && \
    uv export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/requirements.txt && \
    pip3 install -r /tmp/requirements.txt --no-cache-dir
RUN cd CosyVoice/runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto