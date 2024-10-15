FROM osrf/ros:humble-desktop-full

RUN rm /bin/sh && ln -s /bin/bash /bin/sh 

RUN apt-get update && apt-get upgrade -y 

RUN apt-get install -y lsb-release wget gnupg libabsl-dev libgoogle-glog-dev

RUN sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null 

RUN apt-get update && apt-get install -y \
    libgz-math7-dev \
    libgz-transport12-dev libgz-sim7-dev 

RUN git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver && \
    cd ceres-solver && \
    git reset --hard 6a74af202d83cf31811ea17dc66c74d03b89d79e && \
    cmake . && \
    make && \
    sudo make install && \
    cd .. 

RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && \
    sed -i 's/cmake_minimum_required(VERSION [0-9]\+\.[0-9]\+)/cmake_minimum_required(VERSION 3.21)/' CMakeLists.txt && \
    cmake . && \
    make && \
    make install

    


