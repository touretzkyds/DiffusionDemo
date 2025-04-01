"""
Interpolation pipeline for the Diffusion Demo application.

This module implements text prompt interpolation, allowing users to see how images
transform when gradually changing from one text prompt to another. It creates
a smooth animation that transitions between two concepts while maintaining the
same initial latent noise, demonstrating the semantic navigation capabilities
of the text encoder.
"""

import torch
import gradio as gr
from src.util.base import *
from src.util.params import *


def interpolate_prompts(promptA, promptB, num_interpolation_steps):
    """
    Create a series of interpolated embeddings between two text prompts.
    
    This function generates intermediate text embeddings by linearly interpolating
    between the embeddings of two prompts. The interpolation happens in the CLIP
    embedding space, producing a smooth transition between concepts.
    
    Args:
        promptA (str): Starting text prompt
        promptB (str): Ending text prompt
        num_interpolation_steps (int): Number of interpolation steps including endpoints
        
    Returns:
        list: List of interpolated text embeddings
    """
    # Convert both text prompts to CLIP embeddings
    text_embeddingsA = get_text_embeddings(promptA)
    text_embeddingsB = get_text_embeddings(promptB)

    # Initialize list to store interpolated embeddings
    interpolated_embeddings = []

    # Generate a series of embeddings from promptA to promptB
    for i in range(num_interpolation_steps):
        # Calculate interpolation factor (0.0 at start, 1.0 at end)
        alpha = i / num_interpolation_steps
        
        # Linear interpolation: embedding = (1-alpha)*embeddingA + alpha*embeddingB
        # torch.lerp implements this formula efficiently
        interpolated_embedding = torch.lerp(text_embeddingsA, text_embeddingsB, alpha)
        
        # Add the interpolated embedding to our collection
        interpolated_embeddings.append(interpolated_embedding)

    return interpolated_embeddings


def display_interpolate_images(
    seed, promptA, promptB, num_inference_steps, num_images, progress=gr.Progress(), request: gr.Request = None
):
    """
    Generate a series of images interpolating between two text prompts.
    
    This function creates a sequence of images that smoothly transition from one prompt
    to another. Instead of changing the latent noise, it interpolates between the text
    embeddings while keeping the initial noise constant. This demonstrates how the model
    can navigate between different concepts based solely on text guidance.
    
    Args:
        seed (int): Random seed for reproducibility
        promptA (str): Starting text prompt
        promptB (str): Ending text prompt
        num_inference_steps (int): Number of denoising steps for each image
        num_images (int): Number of interpolation steps (intermediate images)
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing the sequence of interpolated images,
            str: Path to generated GIF animation,
            str: Path to ZIP file containing images and config,
            str: Starting prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Generate a single latent vector using the seed - this will be used for all images
    # This ensures that the only difference between images is the text embedding
    latents = generate_latents(seed)
    
    # Add 2 to include the start and end points in the interpolation
    num_images = num_images + 2
    
    # Generate the interpolated text embeddings between the two prompts
    text_embeddings = interpolate_prompts(promptA, promptB, num_images)
    
    # Initialize the list to store generated images
    images = []
    
    # Initialize progress bar
    progress(0)

    # Generate an image for each interpolated embedding
    for i in range(num_images):
        # Update progress indicator
        progress(i / num_images)
        
        # Generate image using the fixed latent noise but with the current interpolated embedding
        # This isolates the effect of changing the text guidance while keeping noise constant
        image = generate_images(latents, text_embeddings[i], num_inference_steps)
        
        # Add the image to our collection with step number as label
        images.append((image, "{}".format(i + 1)))

    # Update progress and indicate we're preparing exports
    progress(1, desc="Exporting as gif")

    # Create metadata for configuration storage
    fname = "interpolate"
    tab_config = {
        "Tab": "Interpolate",
        "First Prompt": promptA,
        "Second Prompt": promptB,
        "Number of Interpolation Steps": num_images,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    
    # Export all images as a ZIP file for downloading
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    
    # Create an animated GIF with reverse playback (forward then backward)
    # This makes the animation loop smoothly instead of jumping back to start
    gif_path = export_as_gif(images, filename="interpolate.gif", reverse=True, request=request)
    
    # Return the gallery of images, paths to the exports, and prompts for UI display
    # Propagate prompts to other tabs for user convenience
    return [gr.Gallery(label="Images", value=images), gif_path, zip_path] + [promptA] * 8

# Define which functions from this module should be importable elsewhere
__all__ = ["display_interpolate_images"]