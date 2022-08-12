# base mirror
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

# setup the directory for installing other softwares
WORKDIR /softwares
RUN cd /softwares
ENV PATH "/softwares/local/bin:${PATH}"
ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/:/softwares/local/lib:${LD_LIBRARY_PATH}"
RUN mkdir -p /softwares/local/bin /softwares/local/lib

# copy extras files for building the docker image
ADD common /softwares/extras

# install Miniconda
RUN bash /softwares/extras/Miniconda3-latest-Linux-x86_64.sh -b -p /softwares/miniconda
ENV PATH "/softwares/miniconda/bin:${PATH}"
RUN pip install --upgrade pip
# RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple

# install CUDA-Toolkit-11.1
#RUN conda install -c nvidia cudatoolkit=11.0  # ?? why not 11.1

# install Python dependencies
COPY environment.yml .
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y git
RUN conda update -n base conda \
    && conda env create -f environment.yml  \
    && source activate ImDrug \
    && pip install git+https://github.com/bp-kelley/descriptastorus

# provide defaults for the executing container
RUN echo "source activate ImDrug" >> ~/.bashrc
ENV PATH "/softwares/miniconda/envs/ImDrug/bin:${PATH}"
CMD [ "/bin/bash" ]
