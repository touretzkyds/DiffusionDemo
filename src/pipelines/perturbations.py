import torch
import numpy as np
import gradio as gr
from src.util.base import *
from src.util.params import *

def display_spread_images(prompt, seed, num_inference_steps, num_images, differentiation, progress=gr.Progress()):
    text_embeddings = get_text_embeddings(prompt)

    latents_x = generate_latents(seed)
    scale_x = torch.cos(torch.linspace(0, 2, num_images)*torch.pi*differentiation/4).to(torch_device)
    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    
    progress(0)
    images = []
    images.append((generate_images(latents_x, text_embeddings, num_inference_steps),"{}".format(0)))

    for i in range(num_images):  
        np.random.seed(i) 
        progress(i/(num_images))
        latents_y = generate_latents(np.random.randint(0, 100000))
        scale_y = torch.sin(torch.linspace(0, 2, num_images)*torch.pi*differentiation/4).to(torch_device)
        noise_y = torch.tensordot(scale_y, latents_y, dims=0)

        noise = noise_x + noise_y
        image = generate_images(noise[num_images-1], text_embeddings, num_inference_steps)
        images.append((image, "{}".format(i+1)))

    return images   

__all__ = [
    "display_spread_images"
]