import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *


def display_perturb_images(
    prompt,
    seed,
    num_inference_steps,
    num_images,
    perturbation_size,
    progress=gr.Progress(),
    request: gr.Request = None
):
    text_embeddings = get_text_embeddings(prompt)

    latents_x = generate_latents(seed)
    scale_x = torch.cos(
        torch.linspace(0, 2, num_images) * torch.pi * perturbation_size / 4
    ).to(torch_device)
    noise_x = torch.tensordot(scale_x, latents_x, dims=0)

    progress(0)
    images = []
    images.append(
        (
            generate_images(latents_x, text_embeddings, num_inference_steps),
            "{}".format(1),
        )
    )

    for i in range(num_images):
        np.random.seed(i)
        progress(i / (num_images))
        latents_y = generate_latents(np.random.randint(0, 100000))
        scale_y = torch.sin(
            torch.linspace(0, 2, num_images) * torch.pi * perturbation_size / 4
        ).to(torch_device)
        noise_y = torch.tensordot(scale_y, latents_y, dims=0)

        noise = noise_x + noise_y
        image = generate_images(
            noise[num_images - 1], text_embeddings, num_inference_steps
        )
        images.append((image, "{}".format(i + 2)))

    fname = "perturbations"
    tab_config = {
        "Tab": "Perturbations",
        "Prompt": prompt,
        "Number of Perturbations": num_images,
        "Perturbation Size": perturbation_size,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    return images, zip_path


__all__ = ["display_perturb_images"]
