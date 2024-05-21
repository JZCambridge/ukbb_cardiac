# FROM tensorflow/tensorflow:latest-gpu

# RUN apt-get update && apt-get install -y build-essential git libjpeg-dev && \
#     apt-get install -y vim tmux curl

# WORKDIR /root
# COPY . /root/UKBBC

# # install python enviroment
# COPY ./requirements.txt /root/UKBBC/requirements.txt
# RUN pip install --upgrade pip setuptools && pip install -r requirements.txt

# # Make port 6666 available to the world outside this container (if needed)
# EXPOSE 6666

# # Set the entrypoint
# ENTRYPOINT [ "python3" ]


# NEED tensor flow 1.x version
FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install necessary packages and add NVIDIA's GPG key
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gnupg2 \
    git \
    libjpeg-dev \
    vim \
    tmux \
    curl \
    ca-certificates \
    libgl1-mesa-glx && \
    apt-get update

# Set the working directory in the container
WORKDIR /root/UKBBC

# Copy the requirements file into the container at /root/UKBBC
COPY requirements.txt .

# Upgrade pip and setuptools and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip setuptools && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to /root/UKBBC
COPY . .

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/root/UKBBC/workspaces/ukbb_cardiac/:"${PYTHONPATH}"

# Make port 6666 available to the world outside this container (if needed)
EXPOSE 6666

# Set the entrypoint
ENTRYPOINT ["python3"]


