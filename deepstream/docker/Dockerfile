FROM nvcr.io/nvidia/deepstream:6.1.1-devel

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

RUN wget \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.4/pyds-1.1.4-py3-none-linux_x86_64.whl && \
    pip3 install ./pyds-1.1.4-py3-none-linux_x86_64.whl

RUN pip install jupyter

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/