"""
Latent space regional modification (poking) pipeline for the Diffusion Demo.

This module implements functionality to modify a specific region of the latent space,
demonstrating how local changes affect the generated image. By 'poking' a small region
of the latent noise, users can observe how modifications in one part of the latent space
can cause both local and global changes in the resulting image, illustrating the complex 
relationship between latent representations and visual features.
"""

import os
import gradio as gr
import torch
import numpy as np
from src.util.base import *
from src.util.params import *
from PIL import Image, ImageDraw, ImageOps
from src.util.session import session_manager


def visualize_poke(
    pokeX, pokeY, pokeHeight, pokeWidth, seed=14, imageHeight=imageHeight, imageWidth=imageWidth, request: gr.Request = None
):
    """
    Visualize the region being modified in both original and poked images.
    
    This function creates noise visualizations with a highlighted region to help users
    understand which part of the latent space is being altered. It visualizes
    the region on both the original and modified noise visualizations, as well as the actual images.
    
    Args:
        pokeX (int): X-coordinate of the center of the poke region (in latent space)
        pokeY (int): Y-coordinate of the center of the poke region (in latent space)
        pokeHeight (int): Height of the poke region (in latent space)
        pokeWidth (int): Width of the poke region (in latent space)
        seed (int): Random seed for noise generation (matched with user's selection)
        imageHeight (int): Height of the output image
        imageWidth (int): Width of the output image
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        tuple: (
            PIL.Image: Noise visualization with poke region highlighted in yellow,
            PIL.Image: Different noise visualization with poke region highlighted in yellow, 
            PIL.Image: Original generated image with poke region highlighted,
            PIL.Image: Modified generated image with poke region highlighted
        )
    """
    # Check if the poke region extends outside the valid latent space boundaries
    # Note: The latent space is 1/8 the size of the pixel space
    if (
        (pokeX - pokeWidth // 2 < 0)
        or (pokeX + pokeWidth // 2 > imageWidth // 8)
        or (pokeY - pokeHeight // 2 < 0)
        or (pokeY + pokeHeight // 2 > imageHeight // 8)
    ):
        # Warn the user if the region goes out of bounds
        gr.Warning("Modification outside image")
    
    # Calculate rectangle coordinates in pixel space (×8 from latent space)
    shape = [
        (pokeX * 8 - pokeWidth * 8 // 2, pokeY * 8 - pokeHeight * 8 // 2),  # Top-left corner
        (pokeX * 8 + pokeWidth * 8 // 2, pokeY * 8 + pokeHeight * 8 // 2),  # Bottom-right corner
    ]

    # Generate random noise images for visualization using the user's seed
    torch.manual_seed(seed)  # Use user's seed for first noise visualization
    noise1 = torch.randn(1, 3, imageHeight, imageWidth)
    noise1 = (noise1 * 0.1 + 0.5).clamp(0, 1)  # Scale to [0.4, 0.6] for gentle contrast
    noise1_np = noise1[0].permute(1, 2, 0).numpy()
    
    # Convert to grayscale by averaging RGB channels
    grayscale_noise1 = np.mean(noise1_np, axis=2)
    grayscale_noise1 = np.stack([grayscale_noise1] * 3, axis=2)
    noise1_image = Image.fromarray((grayscale_noise1 * 255).astype(np.uint8))
    
    # Use seed+1 for the second noise visualization
    torch.manual_seed(seed + 1)
    noise2 = torch.randn(1, 3, imageHeight, imageWidth)
    noise2 = (noise2 * 0.1 + 0.5).clamp(0, 1)  # Scale to [0.4, 0.6] for gentle contrast
    noise2_np = noise2[0].permute(1, 2, 0).numpy()
    
    # Convert to grayscale by averaging RGB channels
    grayscale_noise2 = np.mean(noise2_np, axis=2)
    grayscale_noise2 = np.stack([grayscale_noise2] * 3, axis=2)
    noise2_image = Image.fromarray((grayscale_noise2 * 255).astype(np.uint8))
    
    # Get the session directory for storing/retrieving images
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    original_path = session_dir / "original.png"
    poked_path = session_dir / "poked.png"

    # Try to load existing images if available, otherwise use blank images
    blank = Image.new("RGB", (imageWidth, imageHeight))
    if original_path.exists() and poked_path.exists():
        oImg = Image.open(original_path)
        pImg = Image.open(poked_path)
    else:
        oImg = blank.copy()
        pImg = blank.copy()

    # Create drawing objects for all images
    noise1_draw = ImageDraw.Draw(noise1_image)
    noise2_draw = ImageDraw.Draw(noise2_image)
    oRec = ImageDraw.Draw(oImg)
    pRec = ImageDraw.Draw(pImg)

    # Create a yellow tinted overlay for the poke region
    # Extract the region to be tinted from both noise images
    poke_region1 = noise1_image.crop((shape[0][0], shape[0][1], shape[1][0], shape[1][1]))
    poke_region2 = noise2_image.crop((shape[0][0], shape[0][1], shape[1][0], shape[1][1]))
    
    # Create yellow tint (add a yellow cast while preserving texture)
    yellow_tint = np.array([255, 255, 180])  # Light yellow color
    
    # Apply yellow tint to the poke regions
    poke_region1_np = np.array(poke_region1)
    poke_region1_np = poke_region1_np * 0.7 + yellow_tint * 0.3  # Blend with yellow
    poke_region1 = Image.fromarray(poke_region1_np.astype(np.uint8))
    
    poke_region2_np = np.array(poke_region2)
    poke_region2_np = poke_region2_np * 0.7 + yellow_tint * 0.3  # Blend with yellow
    poke_region2 = Image.fromarray(poke_region2_np.astype(np.uint8))
    
    # Paste the tinted regions back into the noise images
    noise1_image.paste(poke_region1, (shape[0][0], shape[0][1]))
    noise2_image.paste(poke_region2, (shape[0][0], shape[0][1]))
    
    # Draw the rectangle indicating the modified region on all images
    noise1_draw.rectangle(shape, outline="white")
    noise2_draw.rectangle(shape, outline="white")
    oRec.rectangle(shape, outline="white")
    pRec.rectangle(shape, outline="white")

    # Return all four images: two noise visualizations and two actual images
    return noise1_image, noise2_image, oImg, pImg


def display_poke_images(
    prompt,
    seed,
    num_inference_steps,
    poke=False,
    pokeX=None,
    pokeY=None,
    pokeHeight=None,
    pokeWidth=None,
    intermediate=False,
    progress=gr.Progress(),
    request: gr.Request = None
):
    """
    Generate a pair of images - original and with regional latent modification.
    
    This function demonstrates how modifying a specific region of the latent space
    affects the generated image. It creates two images using the same text prompt:
    one with the unmodified latent noise and another where a specific region of the 
    latent noise has been replaced with different noise. This shows how local changes
    in latent space can produce both local and non-local changes in the image.
    
    Args:
        prompt (str): Text prompt to guide image generation
        seed (int): Random seed for reproducibility
        num_inference_steps (int): Number of denoising steps for each image
        poke (bool): Whether to perform the regional modification
        pokeX (int): X-coordinate of the center of the poke region
        pokeY (int): Y-coordinate of the center of the poke region
        pokeHeight (int): Height of the poke region
        pokeWidth (int): Width of the poke region
        intermediate (bool): Whether to return intermediate denoising steps
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        tuple: (
            PIL.Image or list: Original image(s),
            PIL.Image or list or None: Modified image(s) or None if poke=False
        )
    """
    # Get text embeddings for the prompt
    text_embeddings = get_text_embeddings(prompt)
    
    # Generate both original and modified latent vectors
    # If poke=False, modified_latents will be None
    latents, modified_latents = generate_modified_latents(
        poke, seed, pokeX, pokeY, pokeHeight, pokeWidth
    )

    # Start progress tracking
    progress(0)
    
    # Generate the original image using the unmodified latents
    images = generate_images(
        latents, text_embeddings, num_inference_steps, intermediate=intermediate
    )

    # Get session directory for storing images
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    
    # Save the original image if not returning intermediates
    if not intermediate:
        images.save(session_dir / "original.png")

    # Generate the modified image if poke=True
    if poke:
        # Update progress to halfway point
        progress(0.5)
        
        # Generate image with the modified latents
        modImages = generate_images(
            modified_latents,
            text_embeddings,
            num_inference_steps,
            intermediate=intermediate,
        )

        # Save the modified image if not returning intermediates
        if not intermediate:
            modImages.save(session_dir / "poked.png")
    else:
        # No modified image if poke=False
        modImages = None

    # Return both original and modified images
    return images, modImages


# Define which functions from this module should be importable elsewhere
__all__ = ["display_poke_images", "visualize_poke"]
