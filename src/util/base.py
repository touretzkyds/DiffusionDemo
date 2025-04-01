"""
Core utility functions for the Diffusion Demo application.

This module provides essential functionality for generating, processing, and manipulating
text-to-image diffusion models. It includes functions for handling text embeddings, 
latent space operations, image generation, safety checking, and file management.
"""

import os
import json
import torch
import numpy as np
import gradio as gr
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from src.util.params import *
from src.util.clip_config import *
import matplotlib.pyplot as plt
from src.util.params import bad_concepts
from src.util.session import session_manager
from diffusers.image_processor import VaeImageProcessor

def get_text_embeddings(
    prompt,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    torch_device=torch_device,
    batch_size=1,
    negative_prompt=negative_prompt,
):
    """
    Convert text prompts to embeddings used by the diffusion model.
    
    Args:
        prompt (str): The text prompt to generate an image from
        tokenizer: The tokenizer to process the text
        text_encoder: The encoder to create embeddings
        torch_device (str): Device to run computation on ('cuda' or 'cpu')
        batch_size (int): Number of images to generate
        negative_prompt (str): Text describing what to exclude from the image
        
    Returns:
        torch.Tensor: Combined embeddings of the prompt and negative prompt
    """
    # Process the positive prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    
    # Process the negative prompt
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [negative_prompt] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    
    # Concatenate negative and positive embeddings for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


def generate_latents(
    seed,
    height=imageHeight,
    width=imageWidth,
    torch_device=torch_device,
    unet=unet,
    batch_size=1,
):
    """
    Generate initial random latent vectors for the diffusion process.
    
    Args:
        seed (int): Random seed for reproducibility
        height (int): Height of the target image
        width (int): Width of the target image
        torch_device (str): Device to run computation on
        unet: U-Net model configuration for determining latent dimensions
        batch_size (int): Number of images to generate
        
    Returns:
        torch.Tensor: Initial latent vectors
    """
    # Create deterministic random generator
    generator = torch.Generator().manual_seed(int(seed))

    # Generate random latent vectors at 1/8 of the image size (VAE downsampling)
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(torch_device)

    return latents


def generate_modified_latents(
    poke,
    seed,
    pokeX=None,
    pokeY=None,
    pokeHeight=None,
    pokeWidth=None,
    imageHeight=imageHeight,
    imageWidth=imageWidth,
):
    """
    Generate latents with optional regional modification (poking).
    
    This creates a latent vector where a specific region is replaced with noise from a different seed,
    allowing for targeted perturbation of specific image regions.
    
    Args:
        poke (bool): Whether to perform regional modification
        seed (int): Random seed for the base latent vector
        pokeX (int): X-coordinate of the center of the region to modify
        pokeY (int): Y-coordinate of the center of the region to modify
        pokeHeight (int): Height of the region to modify
        pokeWidth (int): Width of the region to modify
        imageHeight (int): Height of the target image
        imageWidth (int): Width of the target image
        
    Returns:
        tuple: (original_latents, modified_latents) - If poke is False, modified_latents is None
    """
    # Generate the original latent vector
    original_latents = generate_latents(seed, height=imageHeight, width=imageWidth)
    
    if poke:
        # Create a new random seed based on the original seed
        np.random.seed(seed)
        poke_latents = generate_latents(
            np.random.randint(0, 100000), height=pokeHeight * 8, width=pokeWidth * 8
        )

        # Calculate the top-left corner of the region to modify
        x_origin = pokeX - pokeWidth // 2
        y_origin = pokeY - pokeHeight // 2

        # Clone the original latents and replace the specified region
        modified_latents = original_latents.clone()
        modified_latents[
            :, :, y_origin : y_origin + pokeHeight, x_origin : x_origin + pokeWidth
        ] = poke_latents
    else:
        modified_latents = None

    return original_latents, modified_latents


def convert_to_pil_image(image):
    """
    Convert a tensor image to a PIL image.
    
    Args:
        image (torch.Tensor): Image tensor in the range [-1, 1]
        
    Returns:
        PIL.Image: The converted PIL image
    """
    # Scale from [-1, 1] to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # Convert to numpy and change dimension order
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    # Convert to uint8 range [0, 255]
    images = (image * 255).round().astype("uint8")
    
    # Convert to PIL image(s)
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

@torch.no_grad()
def custom_safety_check(clip_input, images, filter_strength=-0.1):
    """
    Custom implementation of NSFW content detection using CLIP embeddings.
    
    Args:
        clip_input (torch.Tensor): CLIP model input
        images: The image(s) to check
        filter_strength (float): Adjustment to the NSFW detection threshold (-0.1 is less strict)
        
    Returns:
        tuple: (filtered_images, has_nsfw_concepts) - Images with NSFW content replaced by zeros
    """
    checker = safety_checker
    
    with torch.amp.autocast(device_type="cuda"):
        # Get image embeddings from CLIP model
        pooled_output = checker.vision_model(clip_input)[1] 
        image_embeds = checker.visual_projection(pooled_output)

        # Calculate cosine distances to known NSFW concept embeddings
        cos_dist = cosine_distance(image_embeds, checker.concept_embeds).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"concept_scores": {}, "bad_concepts": []}
            adjustment = filter_strength  # Use negative value to make filter weaker

            # Check each concept's score against its threshold
            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = checker.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)
                    print("NSFW concept found:", bad_concepts[concet_idx])

            result.append(result_img)

        # Determine which images contain NSFW content
        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        # Zero out NSFW images
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images):
                    images[idx] = torch.zeros_like(images[idx])
                else:
                    images[idx] = np.zeros(images[idx].shape)
        
        if not any(has_nsfw_concepts):
            print("No NSFW found in the image")

        return images, has_nsfw_concepts

def cosine_distance(image_embeds, text_embeds):
    """
    Calculate cosine similarity between image and text embeddings.
    
    Args:
        image_embeds (torch.Tensor): Image embeddings
        text_embeds (torch.Tensor): Text embeddings
        
    Returns:
        torch.Tensor: Cosine similarity matrix
    """
    # Normalize embeddings to unit length
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    
    # Calculate cosine similarity (dot product of normalized vectors)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

def run_safety_check(image, filter_strength=0.0):
    """
    Run the safety check pipeline on an image.
    
    This function prepares the image, runs the safety checker, and handles the results.
    
    Args:
        image: The image to check
        filter_strength (float): Adjustment to the NSFW detection threshold
        
    Returns:
        tuple: (filtered_image, has_nsfw_concept) - Image with NSFW content replaced
    """
    # Prepare image for safety checking
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    
    if torch.is_tensor(image):
        feature_extractor_input = image_processor.postprocess(image, output_type="pil")
    else:
        feature_extractor_input = image_processor.numpy_to_pil(image)
    
    # Run the safety checker
    safety_checker_input = feature_extractor(feature_extractor_input, return_tensors="pt").to(device=torch_device)
    image, has_nsfw_concept = custom_safety_check(safety_checker_input.pixel_values, image, filter_strength)
    
    # Post-process the result
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = image_processor.postprocess(image, output_type="latent", do_denormalize=do_denormalize)

    # Replace NSFW images with black images
    if any(has_nsfw_concept):
        image = torch.full_like(image, -1.0)
        gr.Warning(
            "Potential NSFW content was detected in one or more images. A black image will be returned instead."
            " Try again with a different prompt and/or seed."
        )
    return image, has_nsfw_concept

def generate_images(
    latents,
    text_embeddings,
    num_inference_steps,
    unet=unet,
    guidance_scale=guidance_scale,
    vae=vae,
    scheduler=scheduler,
    intermediate=False,
    progress=gr.Progress(),
):
    """
    Generate images using the diffusion model denoising process.
    
    Args:
        latents (torch.Tensor): Initial latent vectors
        text_embeddings (torch.Tensor): Text embeddings for guidance
        num_inference_steps (int): Number of denoising steps
        unet: The U-Net model for denoising
        guidance_scale (float): Strength of guidance from the text prompt
        vae: The VAE model for decoding latents to images
        scheduler: The noise scheduler for the diffusion process
        intermediate (bool): Whether to return intermediate images
        progress (gr.Progress): Gradio progress bar
        
    Returns:
        PIL.Image or list: The generated image(s)
    """
    # Set up the noise scheduler
    scheduler = type(scheduler).from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    
    # Scale the latents according to the scheduler
    latents = latents * scheduler.init_noise_sigma
    images = []
    i = 1

    # Denoising loop
    for t in tqdm(scheduler.timesteps):
        # Prepare latent model input (duplicate for classifier-free guidance)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # Perform guidance by combining conditioned and unconditioned predictions
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # Save intermediate images if requested
        if intermediate:
            progress(((1000 - t) / 1000))
            Latents = 1 / 0.18215 * latents  # Scale for VAE decoding
            with torch.no_grad():
                image = vae.decode(Latents).sample
                images.append(image)
        
        # Update latents for next denoising step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        i += 1

    # Process intermediate images if requested
    if intermediate:
        image, nsfw = run_safety_check(images[-1])    
        if any(nsfw):
            images = [(convert_to_pil_image(image), "NSFW")]
        else:
            images = [(convert_to_pil_image(image), "{}".format(i)) for i, image in enumerate(images)]

    # Final image decoding if not returning intermediates
    if not intermediate:
        Latents = 1 / 0.18215 * latents  # Scale for VAE decoding
        with torch.no_grad():
            image = vae.decode(Latents).sample
            image, _ = run_safety_check(image)
        images = convert_to_pil_image(image)

    return images


def get_word_embeddings(
    prompt, tokenizer=tokenizer, text_encoder=text_encoder, torch_device=torch_device
):
    """
    Get CLIP embeddings for a single word or phrase.
    
    Args:
        prompt (str): The text to embed
        tokenizer: The tokenizer to process the text
        text_encoder: The encoder to create embeddings
        torch_device (str): Device to run computation on
        
    Returns:
        numpy.ndarray: Normalized embedding vector
    """
    # Process the text input
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(torch_device)

    # Generate embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0].reshape(1, -1)

    # Normalize to unit length
    text_embeddings = text_embeddings.cpu().numpy()
    return text_embeddings / np.linalg.norm(text_embeddings)


def get_concat_embeddings(names, merge=False):
    """
    Get embeddings for multiple words or phrases, optionally merging them.
    
    Args:
        names (list): List of words/phrases to embed
        merge (bool): Whether to average the embeddings into a single vector
        
    Returns:
        numpy.ndarray: Matrix of embeddings or a single merged embedding
    """
    embeddings = []

    # Get embeddings for each name
    for name in names:
        embedding = get_word_embeddings(name)
        embeddings.append(embedding)

    # Stack into a matrix
    embeddings = np.vstack(embeddings)

    # Average if requested
    if merge:
        embeddings = np.average(embeddings, axis=0).reshape(1, -1)

    return embeddings


def get_axis_embeddings(A, B):
    """
    Create semantic axis embeddings by subtracting pairs of concept embeddings.
    
    Args:
        A (list): List of "positive" concept words
        B (list): List of "negative" concept words
        
    Returns:
        numpy.ndarray: Average difference vector representing a semantic axis
    """
    emb = []

    # Calculate differences between paired concepts
    for a, b in zip(A, B):
        e = get_word_embeddings(a) - get_word_embeddings(b)
        emb.append(e)

    # Stack and average the differences
    emb = np.vstack(emb)
    ax = np.average(emb, axis=0).reshape(1, -1)

    return ax


def calculate_residual(
    axis, axis_names, from_words=None, to_words=None, residual_axis=1
):
    """
    Calculate residual vectors by projecting embeddings onto semantic axes.
    
    This function computes the component of word embeddings that is orthogonal
    to two selected semantic axes.
    
    Args:
        axis: Matrix of semantic axes
        axis_names: Names of the semantic axes
        from_words (list): Source concept words
        to_words (list): Target concept words
        residual_axis (int): Index of the axis to calculate residuals for
        
    Returns:
        numpy.ndarray: Residual embedding vectors
    """
    # Determine which axes to use (exclude the residual axis)
    axis_indices = [0, 1, 2]
    axis_indices.remove(residual_axis)

    # Get embeddings for the first axis
    if axis_names[axis_indices[0]] in axis_combinations:
        fembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[0]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[0]]] = from_words + to_words
        fembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    # Get embeddings for the second axis
    if axis_names[axis_indices[1]] in axis_combinations:
        sembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[1]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[1]]] = from_words + to_words
        sembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    # Project onto the first axis
    fprojections = fembeddings @ axis[axis_indices[0]].T
    # Project onto the second axis
    sprojections = sembeddings @ axis[axis_indices[1]].T

    # Remove the projection onto the first axis
    partial_residual = fembeddings - (fprojections.reshape(-1, 1) * fembeddings)
    # Remove the projection onto the second axis
    residual = partial_residual - (sprojections.reshape(-1, 1) * sembeddings)

    return residual


def calculate_step_size(num_images, start_degree_circular, end_degree_circular):
    """
    Calculate the step size for circular interpolation.
    
    Args:
        num_images (int): Number of images to generate
        start_degree_circular (float): Starting angle in degrees
        end_degree_circular (float): Ending angle in degrees
        
    Returns:
        float: Step size in degrees
    """
    return (end_degree_circular - start_degree_circular) / (num_images)


def generate_seed_vis(seed):
    """
    Generate a visualization of the random seed.
    
    Args:
        seed (int): The random seed to visualize
        
    Returns:
        matplotlib.pyplot: Plot object with the seed visualization
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate a random vector to visualize
    emb = np.random.rand(15)
    
    # Create a visualization
    plt.close()
    plt.switch_backend("agg")
    plt.figure(figsize=(10, 0.5))
    plt.imshow([emb], cmap="viridis")
    plt.axis("off")
    return plt


def export_as_gif(images, filename="output.gif", duration=500, reverse=False, request: gr.Request = None):
    """
    Export a sequence of images as an animated GIF.
    
    Args:
        images (list): List of images or image tuples to export
        filename (str): Name of the output file
        duration (int): Duration of each frame in milliseconds
        reverse (bool): Whether to append reversed sequence for smooth looping
        request (gr.Request): Gradio request object containing session information
        
    Returns:
        str: Path to the created GIF file
    """
    # Get session directory
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    gif_path = session_dir / filename
    
    # Extract images from tuples if needed
    pil_images = []
    if isinstance(images, list) and isinstance(images[0], tuple):
        pil_images = [img for img, _ in images]
    else:
        pil_images = images
    
    # Add reversed sequence for smooth looping if requested
    if reverse:
        pil_images = pil_images + pil_images[::-1]
    
    # Save as animated GIF
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    return str(gif_path)


def export_as_zip(images, fname, tab_config, request: gr.Request = None):
    """
    Export images and configuration as a ZIP archive.
    
    Args:
        images: The image(s) to export
        fname (str): Base name for the files
        tab_config (dict): Configuration parameters to save
        request (gr.Request): Gradio request object containing session information
        
    Returns:
        str: Path to the created ZIP file
    """
    # Get session directory
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    
    # Save images
    if isinstance(images, list):
        for i, (image, caption) in enumerate(images):
            image.save(session_dir / f"{fname}_{caption}.png")
    else:
        images.save(session_dir / f"{fname}.png")
    
    # Save configuration
    with open(session_dir / f"{fname}_config.txt", "w") as f:
        json.dump(tab_config, f, indent=4)
    
    # Create ZIP archive
    zip_path = session_dir / f"{fname}.zip"
    os.system(f"cd {session_dir} && zip {fname}.zip {fname}*.png {fname}_config.txt")
    return str(zip_path)


def read_html(file_path):
    """
    Read the contents of an HTML file.
    
    Args:
        file_path (str): Path to the HTML file
        
    Returns:
        str: Contents of the HTML file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def get_user_dir(session_hash):
    """
    Get the main directory for a specific user's session.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        Path: Path object pointing to the user's session directory, or None if no session hash
    """
    if not session_hash:
        return None
    user_dir = session_manager.get_session_path(session_hash)
    print(f"User directory path: {user_dir.absolute()}")
    return user_dir


def get_user_examples_dir(session_hash):
    """
    Get the examples directory for a specific user's session.
    
    Creates the directory if it doesn't exist.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        Path: Path object pointing to the user's examples directory, or None if no session hash
    """
    if not session_hash:
        return None
    examples_dir = session_manager.get_file_path(session_hash, "examples")
    examples_dir.mkdir(exist_ok=True)
    return examples_dir


def get_user_viz_dir(session_hash):
    """
    Get the visualizations directory for a specific user's session.
    
    Creates the directory if it doesn't exist.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        Path: Path object pointing to the user's visualizations directory, or None if no session hash
    """
    if not session_hash:
        return None
    viz_dir = session_manager.get_file_path(session_hash, "visualizations")
    viz_dir.mkdir(exist_ok=True)
    return viz_dir

# Export all public functions
__all__ = [
    "get_text_embeddings",
    "generate_latents",
    "generate_modified_latents",
    "generate_images",
    "get_word_embeddings",
    "get_concat_embeddings",
    "get_axis_embeddings",
    "calculate_residual",
    "calculate_step_size",
    "generate_seed_vis",
    "export_as_gif",
    "export_as_zip",
    "read_html",
    "get_user_dir",
    "get_user_examples_dir",
    "get_user_viz_dir",
]
