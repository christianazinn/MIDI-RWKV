FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget build-essential g++ vim fluidsynth timidity git tmux

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt