import random
from src.util.base import *
from src.util.params import *

def display_spread_images(prompt, seed, num_inference_steps, num_images, differentiation):
    text_embeddings = get_text_embeddings(prompt)
    ilatent = generate_latents(seed)

    images = []
    for i in range(num_images):
        latent = ilatent + differentiation*generate_latents(seed + random.randint(0, 1000))
        image = generate_images(latent, text_embeddings, num_inference_steps)
        images.append((image,i+1))

    return images  
