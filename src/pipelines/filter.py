"""
Filter demonstration pipeline for the Diffusion Demo application.

This module implements functionality to visualize the effect of content filtering
on image generation. It allows users to see how the safety filter evaluates images
against specific concepts and how different thresholds affect filtering decisions.

Reference: https://colab.research.google.com/drive/1TWQae-fBpw7vS7j-N1WAM_30Mq2N80JL
"""

import gradio as gr
import torch
from PIL import ImageDraw
from src.util.base import (
    get_text_embeddings,
    generate_latents,
    generate_images,
    export_as_zip,
    cosine_distance
)
from src.util.params import *

@torch.no_grad()
def check_filter_values(clip_input, filter_concept_embeds):
    """
    Evaluate images against filter concepts using CLIP embeddings.
    Based on the approach from:
    https://colab.research.google.com/drive/1TWQae-fBpw7vS7j-N1WAM_30Mq2N80JL
    """

    # Get image embeddings from CLIP model
    pooled_output = safety_checker.vision_model(clip_input)[1] 
    image_embeds = safety_checker.visual_projection(pooled_output)

    # Calculate cosine distances
    # Shape: (2, 77, 768) -> (1, 768)
    filter_concept_embeds = filter_concept_embeds.reshape(-1, 768).squeeze(0)
    filter_cos_dist = cosine_distance(image_embeds, filter_concept_embeds).cpu().numpy()
    filter_score = filter_cos_dist.mean()
    return filter_score

def display_filter_images(
    prompt, 
    filter_concept, 
    num_images, 
    num_inference_steps, 
    filter_threshold, 
    progress=gr.Progress(), 
    request: gr.Request = None
):
    """
    Generate multiple images and apply content filtering with visualization.
    
    This function demonstrates how content filtering works by generating multiple images
    from the same prompt but with different seeds, then evaluating each against the
    specified filter concept with a user-defined threshold. Images that would be filtered
    out are visually marked to show how the filtering system works.
    
    Args:
        prompt (str): Text prompt to guide image generation
        filter_concept (str): Concept to filter against (e.g., "cat", "blood", etc.)
        num_images (int): Number of images to generate with different seeds
        num_inference_steps (int): Number of denoising steps for each image
        filter_threshold (float): Threshold for the filter (-1.0 to 1.0)
        progress (gr.Progress): Gradio progress bar object
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        list: [
            gr.Gallery: Gallery object containing generated images with filter scores,
            str: Path to ZIP file containing images and config
        ]
    """
    # Get text embeddings for prompt
    text_embeddings = get_text_embeddings(prompt)
    
    # Get embeddings for filter concept
    filter_embed = get_text_embeddings(filter_concept)
    
    # Initialize tracking
    progress(0)
    images_data = []
    filter_scores = []
    
    # Generate and process images
    for i in range(num_images):
        progress(i / num_images)
        
        # Generate image using existing pipeline
        latents = generate_latents(i)
        image = generate_images(latents, text_embeddings, num_inference_steps)
        
        # Pre-process image
        filter_checker_input = pipe.feature_extractor(images=image, return_tensors="pt").to("cuda")

        # Run filter value check
        filter_score = check_filter_values(
            filter_checker_input.pixel_values, 
            filter_embed, 
        )
        
        if filter_score > filter_threshold:
            # Create a copy for marking
            filtered_img = image.copy()
            draw = ImageDraw.Draw(filtered_img)
            
            # Add visual indicators for filtered images
            width, height = filtered_img.size
            line_width = 5
            
            # Draw red X
            draw.line((0, 0, width, height), fill="red", width=line_width)
            draw.line((0, height, width, 0), fill="red", width=line_width)
            
            image = filtered_img
        
        # Store results
        label = f"Score: {filter_score:.3f}"
        images_data.append((image, label))
        filter_scores.append(filter_score)
    
    # Prepare export data
    tab_config = {
        "Tab": "Filter",
        "Prompt": prompt,
        "Filter Concept": filter_concept,
        "Number of Images": num_images,
        "Number of Inference Steps": num_inference_steps,
        "Filter Threshold": filter_threshold,
        "Filter Scores": [float(score) for score in filter_scores],
    }
    
    # Export results using existing function
    zip_path = export_as_zip(images_data, "filter", tab_config, request=request)
    
    return [gr.Gallery(label="Images with Filter Scores", value=images_data), zip_path]

# Define exportable functions
__all__ = ["display_filter_images"]