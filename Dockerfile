# Diffusion Demo Dockerfile
# 
# This Dockerfile builds a container for running the Diffusion Demo application with GPU support.
# It uses PyTorch with CUDA for hardware acceleration and sets up the necessary dependencies
# for running the Gradio-based UI and diffusion model inference.
#
# NOTE: This Dockerfile should be placed outside the DiffusionDemo repository directory.
# The COPY command assumes the DiffusionDemo directory is in the same location as this Dockerfile.
#
# References:
# - https://huggingface.co/spaces/SpacesExamples/Gradio-Docker-Template-nvidia-cuda/blob/main/Dockerfile
# - https://www.gradio.app/guides/deploying-gradio-with-docker

# Base image: PyTorch with CUDA 11.8 and cuDNN 9 support
# This provides the necessary deep learning frameworks and GPU libraries
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Configure Gradio to listen on all network interfaces (0.0.0.0)
# This allows connections from outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0 

# Expose port 7860 for the Gradio web interface
EXPOSE 7860

# Copy the application code into the container
# The entire DiffusionDemo directory is copied to maintain the project structure
COPY DiffusionDemo/ DiffusionDemo/

# Install system dependencies and Python requirements
# This multi-stage RUN command:
#  1. Updates package lists
#  2. Installs curl for network operations
#  3. Installs zip for creating downloadable artifacts
#  4. Cleans up to reduce image size
#  5. Installs Python dependencies from requirements.txt
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r DiffusionDemo/requirements.txt -q

# Set the command to run when the container starts
# This launches the demo application with Python 3
CMD ["python3", "DiffusionDemo/run.py"]

# Sample run command (for reference):
# docker run -p 7860:7860 --rm --runtime=nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface akameswa/diffusion-demo-cuda-slim:latest
# 
# This command:
#  - Maps container port 7860 to host port 7860
#  - Removes the container after it stops (--rm)
#  - Enables NVIDIA GPU support
#  - Mounts the Hugging Face cache directory to persist model downloads
#  - Uses the diffusion-demo-cuda-slim image