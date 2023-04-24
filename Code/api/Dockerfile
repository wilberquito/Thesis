# Use an official Anaconda runtime as a parent image
FROM continuumio/anaconda3:latest

# Libraries inside docker
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the environment.yml file to the container
COPY environment.yml .

# Create a new conda environment using the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment
RUN echo "source activate melanoma_api" > ~/.bashrc

# Copy the package files to the container
COPY vicorobot.pkg /vicorobot.pkg

# Activate the conda environment and install the package within the environment
RUN /bin/bash -c "source activate melanoma_api && cd /vicorobot.pkg/ && pip install ."

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local directory into the container at /app
COPY . /app

# Expose port 8080 for api application
# EXPOSE 8080

# Run the command to start the application
# CMD ["python", "main.py"]

ENTRYPOINT ["conda", "run", "-n", "melanoma_api", "python", "main.py"]
