FROM nvcr.io/nvidia/tensorrt:22.09-py3

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh

RUN bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH}

RUN conda update -y conda

RUN rm Miniconda3-py38_4.12.0-Linux-x86_64.sh

RUN conda install -y jupyter

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c conda-forge

# WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/