import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *


def display_circular_images(
    prompt, seed, num_inference_steps, num_images, degree, progress=gr.Progress()
):
    np.random.seed(seed)
    text_embeddings = get_text_embeddings(prompt)

    latents_x = generate_latents(seed)
    latents_y = generate_latents(seed * np.random.randint(0, 100000))

    scale_x = torch.cos(
        torch.linspace(0, 2, num_images) * torch.pi * (degree / 360)
    ).to(torch_device)
    scale_y = torch.sin(
        torch.linspace(0, 2, num_images) * torch.pi * (degree / 360)
    ).to(torch_device)

    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    noise_y = torch.tensordot(scale_y, latents_y, dims=0)

    noise = noise_x + noise_y

    progress(0)
    images = []
    for i in range(num_images):
        progress(i / num_images)
        image = generate_images(noise[i], text_embeddings, num_inference_steps)
        images.append((image, "{}".format(i)))

    progress(1, desc="Exporting as gif")
    export_as_gif(images, filename="circular.gif")

    fname = "circular"
    tab_config = {
        "Tab": "Circular",
        "Prompt": prompt,
        "Number of Steps around the Circle": num_images,
        "Proportion of Circle": degree,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    export_as_zip(images, fname, tab_config)
    return images, "outputs/circular.gif", f"outputs/{fname}.zip"


__all__ = ["display_circular_images"]
