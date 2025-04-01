# Diffusion Demo

## Overview

Diffusion Demo is an educational web application designed to help users understand and explore the inner workings of text-to-image diffusion models. This interactive tool provides a step-by-step exploration of various aspects of stable diffusion models, demonstrating how text prompts are transformed into images and the effect of different parameters on the generation process.

## Live Demo

Try the application online: [diffusiondemo.org](https://diffusiondemo.org)

## Features

The application provides several interactive demonstrations:

### Latent Space Exploration
- **Beginner**: Simple text-to-image generation
- **Denoising**: Visualize the step-by-step denoising process
- **Seeds**: Observe how different random seeds affect image generation
- **Perturbations**: Explore variations around a point in latent space
- **Circular**: Generate a circular path in latent space for smooth transitions
- **Poke**: Modify specific regions of latent space and observe effects
- **Guidance**: Experiment with different guidance scale values
- **Inpainting**: Edit specific parts of an image with text guidance

### CLIP Space Exploration
- **Embeddings**: Visualize and navigate the 3D semantic space of CLIP text embeddings
- **Interpolate**: Generate smooth transitions between two different text prompts
- **Negative**: Learn how negative prompts can be used to avoid certain image characteristics

## Technology

This demo uses:
- **Model**: [Dreamshaper 8](https://huggingface.co/lykon/dreamshaper-8), a fine-tuned version of Stable Diffusion 1.5
- **Frontend**: Gradio for the interactive web interface
- **Backend**: Python with PyTorch and the Diffusers library
- **Visualization**: Plotly for 3D embedding visualization

## Educational Purpose

This application is designed to:
1. Provide intuitive visualizations of complex diffusion concepts
2. Allow hands-on experimentation with model parameters
3. Demonstrate the relationship between text embeddings and generated images
4. Explore the structure of latent space and semantic relationships
5. Serve as a teaching tool for AI/ML courses and workshops

## Development and Deployment

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python DiffusionDemo/run.py`

### Deployment on Jetstream2 Kubernetes

See the [deployment documentation](docs/deployment.md) for more detailed instructions.

## Credits

- **Author**: Adithya Kameswara Rao, Carnegie Mellon University
- **Advisor**: David S. Touretzky, Carnegie Mellon University
- **Funding**: This work was funded by a grant from NEOM Company, and by National Science Foundation award IIS-2112633

## References and Further Reading

- [Cats in Latent Space](https://youtu.be/hb-KT66rCT8?si=RyCQCG7qU-iBugjV) - An intuitive visual explanation of latent diffusion
- [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/) - Interactive explanation of diffusion models
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) - Visual guide to understanding Stable Diffusion
- [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610) - Research paper explaining Stable Diffusion safety filter