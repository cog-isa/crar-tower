FROM tensorflow/tensorflow:1.15.0-gpu

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.6 \
    python3-pip

RUN apt-get install -y python3-pyqt5

RUN git clone https://github.com/agorodetskiy/crar-rllib.git && \
    cd crar-rllib && \
    git checkout master && \
    pip3 install -r requirements.txt && \
    #
    #git clone https://github.com/Unity-Technologies/obstacle-tower-env.git && \
    #cd obstacle-tower-env && \
    #git checkout 008236b0340f2be2d74a7b67bf2e522310e64bcd && \
    #pip3 install -e . && \
    #
    #wget https://storage.googleapis.com/obstacle-tower-build/v4.0/obstacletower_v4.0_linux.zip && \
    #unzip obstacletower_v4.0_linux.zip && \
    cd ..

CMD bash
