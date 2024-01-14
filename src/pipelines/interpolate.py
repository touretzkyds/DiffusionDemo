import torch       
from src.util.base import *
from src.util.params import *                 

def interpolate_prompts(promptA, promptB, num_interpolation_steps):
    text_embeddingsA = get_text_embeddings(promptA)
    text_embeddingsB = get_text_embeddings(promptB)

    interpolated_embeddings = []
    
    for i in range(num_interpolation_steps):
        alpha = i / num_interpolation_steps
        interpolated_embedding = torch.lerp(text_embeddingsA, text_embeddingsB, alpha)
        interpolated_embeddings.append(interpolated_embedding)

    return interpolated_embeddings

def display_interpolate_images(seed, promptA, promptB, num_inference_steps, num_images):
    latents = generate_latents(seed)
    text_embeddings = interpolate_prompts(promptA, promptB, num_images)
    images = []
    for i in range(num_images):   
        image = generate_images(latents, text_embeddings[i], num_inference_steps)
        images.append((image,i+1))
        
    return images
