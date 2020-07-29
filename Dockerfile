FROM tensorflow/tensorflow:1.15.0-gpu

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.7 \
        python3-pip \
        # opencv dependencies
        libsm6 libxext6 libxrender-dev

RUN git clone https://github.com/agorodetskiy/crar-rllib.git \
    && cd crar-rllib \
    && git checkout master \
    && python3 -m pip install -U pip setuptools wheel \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install opencv-python

CMD bash
