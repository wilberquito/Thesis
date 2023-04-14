FROM continuumio/miniconda3

WORKDIR /api

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda activate melanoma_api

# Demonstrate the environment is activated:
RUN echo "Testing if the environment is installed"
RUN python -c "import fastapi"

# Copy the code to container
COPY . .
ENTRYPOINT ["python", "main.py"]
