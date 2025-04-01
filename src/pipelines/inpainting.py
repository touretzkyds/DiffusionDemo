"""
Inpainting pipeline for the Diffusion Demo application.

This module implements functionality for controlled image editing through inpainting.
It allows users to selectively modify portions of an image by drawing a mask over
areas they want to change, and then using text prompts to guide the diffusion model
in generating new content for those masked regions. This demonstrates how diffusion
models can be used not just for creating images from scratch, but also for targeted
image editing and restoration.
"""

import torch
import gradio as gr
from src.util.base import *
from src.util.params import *
from diffusers import AutoPipelineForInpainting

# Initialize the inpainting pipeline from the main diffusion pipeline
# This reuses the same model weights but configures it for inpainting tasks
inpaint_pipe = AutoPipelineForInpainting.from_pipe(pipe).to(torch_device)


def inpaint(dict, num_inference_steps, seed, prompt="", progress=gr.Progress(), request: gr.Request = None):
    """
    Perform inpainting on a masked region of an image.
    
    This function takes a canvas containing both an image and a mask layer, then runs
    the inpainting diffusion model to replace the masked areas with new content guided
    by the text prompt. This demonstrates how diffusion models can selectively modify
    portions of an image while preserving the rest.
    
    Args:
        dict (dict): Dictionary containing the canvas data with 'layers' (mask) and
                    'background' (original image) keys
        num_inference_steps (int): Number of denoising steps for inpainting
        seed (int): Random seed for reproducibility
        prompt (str): Text prompt to guide the inpainting generation
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        tuple: (
            PIL.Image: The inpainted image with masked areas replaced,
            str: Path to ZIP file containing the result images and configuration
        )
    """
    # Initialize progress tracking
    progress(0)
    
    # Extract and prepare the mask and original image from the input dictionary
    # The mask indicates which areas of the image should be replaced (white areas)
    mask = dict["layers"][0].convert("RGB").resize((imageHeight, imageWidth))
    
    # The background is the original image that will be partially modified
    init_image = dict["background"].convert("RGB").resize((imageHeight, imageWidth))
    
    # Run the inpainting pipeline to generate the modified image
    output = inpaint_pipe(
        prompt=prompt,  # Text description for the new content
        negative_prompt=negative_prompt,  # What to avoid in the generation
        image=init_image,  # The original image
        mask_image=mask,  # The mask showing which areas to replace
        guidance_scale=guidance_scale,  # Controls how closely to follow the prompt
        num_inference_steps=num_inference_steps,  # More steps = higher quality but slower
        generator=torch.Generator().manual_seed(seed),  # For reproducible results
    )
    # Update progress to indicate completion of generation
    progress(1)

    # Create metadata for the export
    fname = "inpainting"
    tab_config = {
        "Tab": "Inpainting",
        "Prompt": prompt,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }

    # Prepare images for the gallery and export
    imgs_list = []
    imgs_list.append((output.images[0], "Inpainted Image"))  # The result
    imgs_list.append((mask, "Mask"))  # The mask for reference

    # Export images as a ZIP file for downloading
    zip_path = export_as_zip(imgs_list, fname, tab_config, request=request)
    progress(1.0, desc="Done")
    
    # Return the inpainted image and the ZIP file path
    return output.images[0], zip_path


# Define which functions from this module should be importable elsewhere
__all__ = ["inpaint"]
