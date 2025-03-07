# https://huggingface.co/spaces/SpacesExamples/Gradio-Docker-Template-nvidia-cuda/blob/main/Dockerfile
# https://www.gradio.app/guides/deploying-gradio-with-docker
# docker run -p 7860:7860 --rm --runtime=nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface akameswa/diffusion-demo-cuda-slim:latest
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV GRADIO_SERVER_NAME=0.0.0.0 
EXPOSE 7860

COPY DiffusionDemo/ DiffusionDemo/

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r DiffusionDemo/requirements.txt -q

CMD ["python3", "DiffusionDemo/run.py"]