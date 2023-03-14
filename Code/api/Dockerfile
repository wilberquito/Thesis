FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "melanoma_api", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure fastapi is installed:"
RUN python -c "import fastapi"

# Copy the code to container
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "melanoma_api", "python", "run.py"]