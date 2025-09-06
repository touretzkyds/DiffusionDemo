"""
Seed variation pipeline for the Diffusion Demo application.

This module implements functionality to visualize how different random seeds
produce varying images from the same text prompt. By generating multiple images
with sequential seed values starting from a user-specified seed, users can observe 
the diversity of possible outcomes from the diffusion model and understand how the 
initial random noise significantly influences the generated result.
"""

import gradio as gr
from src.util.base import *
from src.util.params import *


def display_seed_images(
    prompt, num_inference_steps, num_images, starting_seed=0, progress=gr.Progress(), request: gr.Request = None
):
    """
    Generate multiple images with different random seeds from the same prompt.
    
    This function demonstrates how varying the random seed while keeping the text prompt
    constant produces different image outputs. Each image uses a sequential seed value
    starting from the specified starting_seed, allowing users to explore the range of 
    possible generations for a given text description and generate multiple batches
    by adjusting the starting seed.
    
    Args:
        prompt (str): Text prompt to guide all image generations
        num_inference_steps (int): Number of denoising steps for each image
        num_images (int): Number of different seeds/images to generate
        starting_seed (int): Starting seed value for the sequence
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing the generated images with seed labels,
            str: Path to ZIP file containing images and configuration,
            str: Prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Get text embeddings for the prompt (used for all images)
    text_embeddings = get_text_embeddings(prompt)

    # List to store generated images with their seed labels
    images = []
    
    # Initialize progress tracking
    progress(0)

    # Generate images with different seeds but same prompt
    for i in range(num_images):
        # Update progress bar
        progress(i / num_images)
        
        # Generate latent vector using the starting seed plus current index
        current_seed = starting_seed + i
        latents = generate_latents(current_seed)
        
        # Generate image from latents and text embeddings
        image = generate_images(latents, text_embeddings, num_inference_steps)
        
        # Add the image with its actual seed number as label
        images.append((image, "{}".format(current_seed)))

    # Create metadata for the exports
    fname = "seeds"
    tab_config = {
        "Tab": "Seeds",
        "Prompt": prompt,
        "Starting Seed": starting_seed,
        "Number of Seeds": num_images,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed Range": f"{starting_seed} to {starting_seed + num_images - 1}",
    }
    
    # Export images as a ZIP file for downloading
    zip_path = export_as_zip(images, fname, tab_config, request=request)

    # Return gallery, ZIP path, and repeat prompt for user convenience
    return [gr.Gallery(label="Images", value=images), zip_path] + [prompt] * 8


# Define which functions from this module should be importable elsewhere
__all__ = ["display_seed_images"]