"""
Guidance scale demonstration pipeline for the Diffusion Demo application.

This module implements functionality to visualize the effect of different guidance scale
values on the same image generation. The guidance scale controls how closely the generated
image follows the text prompt, with higher values enforcing stronger adherence to the prompt
but potentially introducing artifacts at extreme values.
"""

import gradio as gr
from src.util.base import *
from src.util.params import *


def display_guidance_images(
    prompt, seed, num_inference_steps, guidance_values, progress=gr.Progress(), request: gr.Request = None
):
    """
    Generate multiple images using different guidance scale values.
    
    This function demonstrates how the guidance scale parameter affects image generation
    by using the same initial noise and prompt but varying the guidance scale. Higher
    guidance scale values make the image adhere more closely to the text prompt, while
    lower values allow more freedom but may produce less relevant results.
    
    Args:
        prompt (str): Text prompt to guide image generation
        seed (int): Random seed for reproducibility
        num_inference_steps (int): Number of denoising steps
        guidance_values (str): Comma or space separated list of guidance scale values
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing generated images with different guidance scales,
            str: Path to ZIP file containing images and config,
            str: Prompt (repeated 8 times for UI compatibility)
        ]
    """
    # Get text embeddings for the prompt
    text_embeddings = get_text_embeddings(prompt)
    
    # Generate initial latent noise (same for all guidance values)
    latents = generate_latents(seed)

    # Initialize progress tracking
    progress(0)
    images = []
    
    # Parse guidance values from string input
    guidance_values = guidance_values.replace(",", " ").split()
    num_images = len(guidance_values)

    # Generate an image for each guidance value
    for i in range(num_images):
        # Update progress
        progress(i / num_images)
        
        # Generate image with current guidance scale
        image = generate_images(
            latents,
            text_embeddings,
            num_inference_steps,
            guidance_scale=int(guidance_values[i]),
        )
        
        # Store image with its guidance value as label
        images.append((image, "{}".format(int(guidance_values[i]))))

    # Create metadata for the downloads
    fname = "guidance"
    tab_config = {
        "Tab": "Guidance",
        "Prompt": prompt,
        "Guidance Scale Values": guidance_values,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    
    # Export images as a ZIP file
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    
    # Return gallery, ZIP file path, and repeated prompt for UI
    return [gr.Gallery(label="Images", value=images), zip_path] + [prompt] * 8


# Export the public function from this module
__all__ = ["display_guidance_images"]
