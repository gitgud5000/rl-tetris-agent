# Use the official TensorFlow Docker image with GPU support
FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY entrypoint.sh /
COPY ssh_config.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Add the NVIDIA package repositories public key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


ENV DEBIAN_FRONTEND noninteractive
# Install necessary packages and libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx\
    # build-essential \
    rsync \
    nano \
    # cmake \
    git \
    wget \
    unzip \
    xauth \
    openssh-server \
    x11-apps \
    # pkg-config \
    # python3-dev \
    # python3-pip \
    # python3-opencv \
    # libopencv-dev \
    # libgtk-3-dev \
    # libboost-all-dev \
    && apt-get clean

# Ensure the SSH directory exists and has the correct permissions
RUN mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh \
    && echo "$PUBLIC_KEY" >> authorized_keys \
    && chmod 700 authorized_keys

CMD service ssh start;sleep infinity
RUN chmod +x /ssh_config.sh


# Upgrade pip and install necessary Python packages
RUN pip3 install --upgrade pip
RUN pip3 install \
    tqdm \
    opencv-python 
    # opencv-python-headless \
    # scipy \
    # matplotlib \
    # pillow \
    # scikit-learn \
    # jupyter \
    # pandas

# Install Visual Studio Code
# RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg \
#     && install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ \
#     && sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list' \
#     && apt-get update \
#     && apt-get install -y code \
#     && rm microsoft.gpg
# Set the default command to bash
# CMD ["bash"]
