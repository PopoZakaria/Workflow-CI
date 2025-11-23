# Gunakan miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy seluruh project
COPY . /app

# Buat environment conda
RUN conda env create -f conda.yaml

# Gunakan shell dari conda env
SHELL ["conda", "run", "-n", "retrain_env", "/bin/bash", "-c"]

# Default command jalankan MLflow project
CMD ["python", "modelling.py"]
