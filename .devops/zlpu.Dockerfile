ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt update && apt install -y git build-essential cmake wget xz-utils
RUN apt install libcurl4-openssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev
    