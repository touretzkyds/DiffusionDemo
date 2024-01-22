import torch
from src.util.base import *
from src.util.params import *

def display_spread_images(prompt, seed, num_inference_steps, num_images, differentiation):
    text_embeddings = get_text_embeddings(prompt)
    initial_latent = generate_latents(seed)

    images = []
    images.append((generate_images(initial_latent, text_embeddings, num_inference_steps),0))

    for i in range(num_images + 1):
        final_latent = generate_latents(i + 1)
        latent = torch.lerp(initial_latent, final_latent, differentiation)
        image = generate_images(latent, text_embeddings, num_inference_steps)
        images.append((image, i+1))

    return images 

__all__ = [
    "display_spread_images"
]