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
from src.util.base import *
from src.util.params import *
from PIL import Image, ImageDraw
from src.util.session import session_manager


def visualize_poke(
    pokeX, pokeY, pokeHeight, pokeWidth, imageHeight=imageHeight, imageWidth=imageWidth, request: gr.Request = None
):
    """
    Visualize the region being modified in both original and poked images.
    
    This function draws a rectangle around the modified region to help users
    understand which part of the latent space is being altered. It visualizes
    the region on both the original and modified images for comparison.
    
    Args:
        pokeX (int): X-coordinate of the center of the poke region (in latent space)
        pokeY (int): Y-coordinate of the center of the poke region (in latent space)
        pokeHeight (int): Height of the poke region (in latent space)
        pokeWidth (int): Width of the poke region (in latent space)
        imageHeight (int): Height of the output image
        imageWidth (int): Width of the output image
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        tuple: (
            PIL.Image: Visualization blank image with poke region highlighted,
            PIL.Image: Visualization blank image with poke region highlighted, 
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

    # Create blank images for visualization
    blank1 = Image.new("RGB", (imageWidth, imageHeight))
    blank2 = Image.new("RGB", (imageWidth, imageHeight))
    
    # Get the session directory for storing/retrieving images
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    original_path = session_dir / "original.png"
    poked_path = session_dir / "poked.png"

    # Try to load existing images if available, otherwise use blank images
    if original_path.exists() and poked_path.exists():
        oImg = Image.open(original_path)
        pImg = Image.open(poked_path)
    else:
        oImg = blank1.copy()
        pImg = blank2.copy()

    # Create drawing objects for all images
    blankRec1 = ImageDraw.Draw(blank1)
    blankRec2 = ImageDraw.Draw(blank2)
    oRec = ImageDraw.Draw(oImg)
    pRec = ImageDraw.Draw(pImg)

    # Draw the rectangle indicating the modified region on all images
    blankRec1.rectangle(shape, outline="white")
    blankRec2.rectangle(shape, outline="white")
    oRec.rectangle(shape, outline="white")
    pRec.rectangle(shape, outline="white")

    # Return all four images: two visualization blanks and two actual images
    return blank1, blank2, oImg, pImg


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
