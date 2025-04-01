"""
Circular interpolation pipeline for the Diffusion Demo application.

This module implements a technique to generate a series of images along a circular path
in latent space. It creates a smooth animation that starts and ends at the same point,
allowing for visualization of continuous transformations through the latent space.
"""

import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *


def display_circular_images(
    prompt, seed, num_inference_steps, num_images, start_degree, end_degree, progress=gr.Progress(), request: gr.Request = None
):
    """
    Generate images along a circular path in latent space.
    
    This function creates a series of images by interpolating through a circular path in 
    latent space. It uses two latent vectors as the basis for the circle, with cosine and sine
    functions determining their relative weights at each point along the path. This
    creates a smooth transition that returns to the starting point when the full circle
    is completed.
    
    Args:
        prompt (str): Text prompt to guide image generation
        seed (int): Random seed for reproducibility
        num_inference_steps (int): Number of denoising steps for each image
        num_images (int): Number of images to generate along the path
        start_degree (float): Starting angle in degrees for the circular path
        end_degree (float): Ending angle in degrees for the circular path
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing generated images,
            str: Path to generated GIF animation,
            str: Path to ZIP file containing images and config,
            str: Prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Add one to num_images to include the end point
    num_images += 1
    
    # Get text embeddings for the prompt
    text_embeddings = get_text_embeddings(prompt)

    # Generate two orthogonal latent vectors to define the circular space
    latents_x = generate_latents(seed)  # First base vector using provided seed
    latents_y = generate_latents(seed * np.random.randint(0, 100000))  # Second base vector using derived seed

    # Generate points along a circular path using trigonometric functions
    # Cosine for x-coordinate
    scale_x = torch.cos(
        torch.linspace(start_degree, end_degree, num_images) * torch.pi / 180
    ).to(torch_device)
    
    # Sine for y-coordinate
    scale_y = torch.sin(
        torch.linspace(start_degree, end_degree, num_images) * torch.pi / 180
    ).to(torch_device)

    # Scale the latent vectors by the circular coordinates
    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    noise_y = torch.tensordot(scale_y, latents_y, dims=0)

    # Combine the scaled latent vectors to create points on the circle
    noise = noise_x + noise_y

    # Initialize progress bar
    progress(0)
    images = []
    
    # Generate images for each point on the circular path
    for i in range(num_images):
        # Update progress
        progress(i / num_images)
        
        # Generate image from current latent point
        image = generate_images(noise[i], text_embeddings, num_inference_steps)
        
        # Calculate and store the actual angle for this point (for labeling)
        angle = start_degree + i*(end_degree-start_degree)/(num_images-1)
        images.append((image, str(angle)))

    # Complete progress bar and prepare exports
    progress(1, desc="Exporting as gif")

    # Create configuration for metadata
    fname = "circular"
    tab_config = {
        "Tab": "Circular",
        "Prompt": prompt,
        "Number of Steps around the Circle": num_images,
        "Start Proportion of Circle": start_degree,
        "End Proportion of Circle": end_degree,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    
    # Export results as ZIP file and GIF animation
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    gif_path = export_as_gif(images, filename="circular.gif", request=request)
    
    # Return gallery, file paths, and repeated prompt for UI compatibility
    return [gr.Gallery(label="Images", value=images), gif_path, zip_path] + [prompt] * 8


# Export the public function from this module
__all__ = ["display_circular_images"]
