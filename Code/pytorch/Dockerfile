FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "melanoma_nn", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure pytorch is installed:"
RUN python -c "import torch"

# Copy the code to container
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "melanoma_nn", "python", "run.py"]