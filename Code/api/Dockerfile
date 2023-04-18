FROM continuumio/miniconda3

# Conda requires a shell to run
RUN conda init bash

WORKDIR /api

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment
RUN conda activate melanoma_api

# Demonstrate the environment is activated:
RUN echo "Testing if the environment is installed"
RUN python -c "import fastapi"

# Copy the code to container
COPY . .

# Install vicorobot module
RUN cd vicorobot \
    && pip install vicorobot

ENTRYPOINT ["python", "main.py"]
