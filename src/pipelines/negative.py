"""
Negative prompt demonstration pipeline for the Diffusion Demo application.

This module implements functionality to visualize the effect of negative prompts on
image generation. Negative prompts tell the model what to avoid including in the image,
allowing for more precise control over the generated content. This demonstration
generates paired images - one with just the positive prompt and one with both positive
and negative prompts - to show the difference.
"""

import gradio as gr
from src.util.base import *
from src.util.params import *


def display_negative_images(
    prompt, seed, num_inference_steps, negative_prompt="", progress=gr.Progress(), request: gr.Request = None
):
    """
    Generate two images showing the effect of negative prompts.
    
    This function demonstrates how negative prompts influence image generation
    by creating two images with the same initial noise and positive prompt,
    but one with and one without the negative prompt. This shows how negative
    prompts can help avoid unwanted elements in the generated images.
    
    Args:
        prompt (str): Main text prompt to guide what should appear in the image
        seed (int): Random seed for reproducibility
        num_inference_steps (int): Number of denoising steps
        negative_prompt (str): Text describing what should NOT appear in the image
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            PIL.Image: Image generated without negative prompt,
            PIL.Image: Image generated with negative prompt,
            str: Path to ZIP file containing images and config,
            str: Prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Get text embeddings for the positive prompt alone
    text_embeddings = get_text_embeddings(prompt)
    
    # Get text embeddings combining positive and negative prompts
    text_embeddings_neg = get_text_embeddings(prompt, negative_prompt=negative_prompt)

    # Generate initial latent noise (same for both images to ensure fair comparison)
    latents = generate_latents(seed)

    # Start progress tracking
    progress(0)
    
    # Generate image using just the positive prompt
    images = generate_images(latents, text_embeddings, num_inference_steps)

    # Update progress to halfway point
    progress(0.5)
    
    # Generate image using the same noise but with negative prompt applied
    images_neg = generate_images(latents, text_embeddings_neg, num_inference_steps)

    # Create metadata for the exports
    fname = "negative"
    tab_config = {
        "Tab": "Negative",
        "Prompt": prompt,
        "Negative Prompt": negative_prompt,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }

    # Prepare images for the gallery with labels
    imgs_list = []
    imgs_list.append((images, "Without Negative Prompt"))
    imgs_list.append((images_neg, "With Negative Prompt"))
    
    # Export images as a ZIP file for downloading
    zip_path = export_as_zip(imgs_list, fname, tab_config, request=request)
    
    # Return both images, the ZIP path, and repeat prompt for user convenience
    return [images, images_neg, zip_path] + [prompt] * 8


# Define which functions from this module should be importable elsewhere
__all__ = ["display_negative_images"]
