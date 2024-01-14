from src.util.base import *
from src.util.params import *

def display_seed_images(prompt, num_inference_steps, num_images):
    text_embeddings = get_text_embeddings(prompt)

    images = []
    for i in range(num_images):   
        latents = generate_latents(i)
        image = generate_images(latents, text_embeddings, num_inference_steps)
        images.append((image,i+1))

    return images     

__all__ = [
    "display_seed_images"
]