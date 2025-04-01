"""
Latent space perturbation pipeline for the Diffusion Demo application.

This module implements functionality to visualize how small changes in the latent space
affect the generated images. By exploring different perturbations around a base latent
vector, users can see the variations that exist in a local region of the latent space.
This helps demonstrate how similar yet distinct images cluster in the model's latent space.
"""

import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *


def display_perturb_images(
    prompt,
    seed,
    num_inference_steps,
    num_images,
    perturbation_size,
    progress=gr.Progress(),
    request: gr.Request = None
):
    """
    Generate images with controlled perturbations in the latent space.
    
    This function demonstrates how small variations in latent space produce different but
    related images. It generates a base image plus multiple perturbed variants by combining
    the base latent vector with different orthogonal vectors, scaled by the perturbation size.
    The larger the perturbation size, the more the images will differ from the base image.
    
    Args:
        prompt (str): Text prompt to guide image generation
        seed (int): Random seed for reproducibility of the base latent vector
        num_inference_steps (int): Number of denoising steps for each image
        num_images (int): Number of perturbed images to generate
        perturbation_size (float): Scale of the perturbations (0.0-1.0)
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing the base and perturbed images,
            str: Path to ZIP file containing images and config,
            str: Prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Get text embeddings for the prompt
    text_embeddings = get_text_embeddings(prompt)

    # Generate the base latent vector that all perturbations will be relative to
    latents_x = generate_latents(seed)
    
    # Create a scaling factor for the base latent using a cosine curve
    # This determines how much of the original latent is preserved in each perturbation
    scale_x = torch.cos(
        torch.linspace(0, 2, num_images) * torch.pi * perturbation_size / 4
    ).to(torch_device)
    
    # Combine the scale factors with the base latent to create a tensor of scaled base latents
    noise_x = torch.tensordot(scale_x, latents_x, dims=0)

    # Initialize progress tracking
    progress(0)
    images = []
    
    # Generate and add the base image (unperturbed)
    images.append(
        (
            generate_images(latents_x, text_embeddings, num_inference_steps),
            "{}".format(1),  # Label as image 1
        )
    )

    # Generate each perturbed image by combining the base latent with a new orthogonal component
    for i in range(num_images):
        # Set a new random seed for each perturbation, but deterministically based on index
        np.random.seed(i)
        
        # Update progress
        progress(i / (num_images))
        
        # Generate a random orthogonal latent vector for perturbation
        latents_y = generate_latents(np.random.randint(0, 100000))
        
        # Scale the orthogonal component using a sine curve (orthogonal to cosine)
        # This ensures the perturbation vectors are roughly perpendicular to the base
        scale_y = torch.sin(
            torch.linspace(0, 2, num_images) * torch.pi * perturbation_size / 4
        ).to(torch_device)
        
        # Combine the scale factors with the orthogonal latent
        noise_y = torch.tensordot(scale_y, latents_y, dims=0)

        # Create the perturbed latent by adding both components
        # We take the last element which has maximum perturbation based on the curves
        noise = noise_x + noise_y
        
        # Generate the image from the perturbed latent
        image = generate_images(
            noise[num_images - 1], text_embeddings, num_inference_steps
        )
        
        # Add to our collection with sequential numbering
        images.append((image, "{}".format(i + 2)))

    # Create metadata for the exports
    fname = "perturbations"
    tab_config = {
        "Tab": "Perturbations",
        "Prompt": prompt,
        "Number of Perturbations": num_images,
        "Perturbation Size": perturbation_size,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    
    # Export images as a ZIP file for downloading
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    
    # Return gallery, ZIP path, and repeat prompt for user convenience
    return [gr.Gallery(label="Images", value=images), zip_path] + [prompt] * 8


# Define which functions from this module should be importable elsewhere
__all__ = ["display_perturb_images"]
