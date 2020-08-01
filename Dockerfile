FROM tensorflow/tensorflow:1.15.0-gpu

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    # for conda
    bzip2 ca-certificates libglib2.0-0 \
    # opencv dependencies
    libsm6 libxext6 libxrender-dev \
    # some log sync req
    rsync

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    conda create --name conda-env python=3.7.3

COPY . /crar-rllib

RUN cd crar-rllib \
    && git clone https://github.com/Unity-Technologies/obstacle-tower-env.git \
    && cd obstacle-tower-env \
    && git checkout 008236b0340f2be2d74a7b67bf2e522310e64bcd \
    && wget https://storage.googleapis.com/obstacle-tower-build/v4.0/obstacletower_v4.0_linux.zip \
    && unzip obstacletower_v4.0_linux.zip

# activate env, install both main and env dependencies

CMD bash
