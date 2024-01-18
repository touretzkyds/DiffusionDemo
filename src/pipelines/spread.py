import torch
import random
from src.util.base import *
from src.util.params import *

def display_spread_images(prompt, seed, num_inference_steps, num_images, differentiation):
    text_embeddings = get_text_embeddings(prompt)
    initial_latent = generate_latents(seed)

    images = []
    for i in range(num_images):
        final_latent = generate_latents(random.randint(0, 1000))
        latent = torch.lerp(initial_latent, final_latent, differentiation)
        image = generate_images(latent, text_embeddings, num_inference_steps)
        images.append((image,i+1))

    return images  

__all__ = [
    "display_spread_images"
]