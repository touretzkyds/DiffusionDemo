import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *


def display_circular_images(
    prompt, seed, num_inference_steps, num_images, start_degree, end_degree, progress=gr.Progress(), request: gr.Request = None
):
    np.random.seed(seed)
    num_images += 1
    text_embeddings = get_text_embeddings(prompt)

    latents_x = generate_latents(seed)
    latents_y = generate_latents(seed * np.random.randint(0, 100000))

    scale_x = torch.cos(
        torch.linspace(start_degree, end_degree, num_images) * torch.pi / 180
    ).to(torch_device)
    scale_y = torch.sin(
        torch.linspace(start_degree, end_degree, num_images) * torch.pi / 180
    ).to(torch_device)

    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    noise_y = torch.tensordot(scale_y, latents_y, dims=0)

    noise = noise_x + noise_y

    progress(0)
    images = []
    for i in range(num_images):
        progress(i / num_images)
        image = generate_images(noise[i], text_embeddings, num_inference_steps)
        images.append((image, str(start_degree + i*(end_degree-start_degree)/(num_images-1))))

    progress(1, desc="Exporting as gif")
    # export_as_gif(images, filename="circular.gif")

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
    
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    gif_path = export_as_gif(images, filename="circular.gif", request=request)
    return images, gif_path, zip_path


__all__ = ["display_circular_images"]
