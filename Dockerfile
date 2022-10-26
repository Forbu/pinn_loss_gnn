# dockerfile definition to create a docker image for the application (pinn_loss_gnn)
# jax base image
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

# install python3.9
RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install jax
RUN pip3 install --upgrade pip
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install flax
RUN pip install flax

