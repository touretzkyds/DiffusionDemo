import torch  
import gradio as gr     
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

def display_interpolate_images(seed, promptA, promptB, num_inference_steps, num_images, progress=gr.Progress()):
    latents = generate_latents(seed)
    text_embeddings = interpolate_prompts(promptA, promptB, num_images)
    images = []
    progress(0)
    for i in range(num_images):   
        progress(i/num_images)
        image = generate_images(latents, text_embeddings[i], num_inference_steps)
        images.append((image,"{}".format(i+1)))

    progress(1, desc="Exporting as gif")
    export_as_gif(images, filename="interpolate.gif", reverse=True)

    fname = "interpolate"
    tab_config = {
        "Tab"                                       : "Interpolate",
        "First Prompt"                              : promptA, 
        "Second Prompt"                             : promptB, 
        "Number of Interpolation Steps"             : num_images,
        "Number of Inference Steps per Image"       : num_inference_steps,
        "Seed"                                      : seed,
    }
    export_as_zip(images, fname, tab_config)
    return images, "outputs/interpolate.gif", f"outputs/{fname}.zip"

__all__ = [
    "display_interpolate_images"
]