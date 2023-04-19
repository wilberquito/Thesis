# Use an official Anaconda runtime as a parent image
FROM continuumio/anaconda3:latest

# Libraries inside docker
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the environment.yml file to the container
COPY environment.yml .

# Create a new conda environment using the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local directory into the container at /app
COPY . /app

# Expose port 8000 for web application access
EXPOSE 8000

# Run the command to start the application
CMD ["python", "main.py"]
