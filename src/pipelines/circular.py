import torch
from src.util.base import *
from src.util.params import *

def display_circular_images(prompt, seed, num_inference_steps, num_images, differentiation):
    text_embeddings = get_text_embeddings(prompt)

    latents_x = generate_latents(seed)
    latents_y = generate_latents(seed*180)

    scale_x = torch.cos(torch.linspace(0, 1, 10)*2*torch.pi*differentiation).to(torch_device)
    scale_y = torch.sin(torch.linspace(0, 1, 10)*2*torch.pi*differentiation).to(torch_device)

    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    noise_y = torch.tensordot(scale_y, latents_y, dims=0)

    noise = noise_x + noise_y
    batched_noise = torch.split(noise, num_images)

    images = []
    for i in range(num_images):   
        image = generate_images(batched_noise[0][i], text_embeddings, num_inference_steps)
        images.append((image,i+1))

    return images    

__all__ = [
    "display_circular_images"
]