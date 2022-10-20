FROM nvcr.io/nvidia/tensorrt:22.09-py3

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh

RUN bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH}

RUN conda update -y conda

RUN rm Miniconda3-py38_4.12.0-Linux-x86_64.sh

RUN conda install -y jupyter

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

RUN apt update

# RUN apt upgrade -y

RUN apt install tini

RUN pip install pycuda onnx scikit-image

RUN conda install matplotlib -y

RUN pip install nvidia-pyindex nvidia-tensorrt

RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/