# Use an official Anaconda runtime as a parent image
FROM docker.io/continuumio/anaconda3:latest

# Libraries inside docker
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the environment.yml file to the container
COPY environment.yml .

# Create a new conda environment using the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment
RUN echo "source activate melanoma_api" > ~/.bashrc

# Copy the packages into the container
COPY _pkgs /_pkgs

# Activate the conda environment and install the package within the environment
RUN /bin/bash -c "source activate melanoma_api && \
                  cd /_pkgs/vicorobot.pkg/ && pip install . \
                  cd /_pkgs/modular.pkg/ && pip install ."

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local directory into the container at /app
COPY . /app

# Run api using conda environment
CMD conda run -n melanoma_api python main.py
