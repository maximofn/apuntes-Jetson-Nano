FROM nvcr.io/nvidia/tensorrt:22.09-py3

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
RUN bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
ENV PATH=/root/miniconda3/bin:${PATH}
RUN conda update -y conda
RUN rm Miniconda3-py38_4.12.0-Linux-x86_64.sh

# Intal basic python packages
RUN conda install -y jupyter numpy pandas matplotlib

# Install pytorch
RUN conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

# Tini
RUN apt update
# RUN apt upgrade -y
RUN apt install tini

RUN pip install pycuda onnx scikit-image

RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt==8.4.3.1
# RUN pip install torch-tensorrt==1.2.0 -f https://github.com/pytorch/tensorrt/releases

# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]

# WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/