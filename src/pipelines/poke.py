import gradio as gr
from src.util.base import *
from src.util.params import *  
from PIL import Image, ImageDraw

def visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, imageHeight=imageHeight, imageWidth=imageWidth):
    shape = [(pokeX - pokeWidth // 2, pokeY - pokeHeight // 2), (pokeX + pokeWidth // 2, pokeY + pokeHeight // 2)] 
    img = Image.new("RGB", (imageHeight, imageWidth))
    rec = ImageDraw.Draw(img) 
    rec.rectangle(shape, outline ="white") 
    return img

def display_poke_images(prompt, seed, num_inference_steps, poke=False, pokeX=None, pokeY=None, pokeHeight=None, pokeWidth=None, intermediate=False, progress=gr.Progress()):
    text_embeddings = get_text_embeddings(prompt)
    latents, modified_latents = generate_modified_latents(poke, seed, pokeX, pokeY, pokeHeight, pokeWidth)

    progress(0)
    images = generate_images(latents, text_embeddings, num_inference_steps, intermediate=intermediate)

    if poke:
        modImages = generate_images(modified_latents, text_embeddings, num_inference_steps, intermediate=intermediate)
    else:    
        modImages = None
    
    return images, modImages

__all__ = [
    "display_poke_images", 
    "visualize_poke"
]