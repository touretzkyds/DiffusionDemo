import os
import gradio as gr
from src.util.base import *
from src.util.params import *  
from PIL import Image, ImageDraw

def visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, imageHeight=imageHeight, imageWidth=imageWidth):
    if ((pokeX - pokeWidth // 2 < 0) or (pokeX + pokeWidth // 2 > imageWidth//8) or (pokeY - pokeHeight // 2 < 0) or (pokeY + pokeHeight // 2 > imageHeight//8)):
        gr.Warning("Modification outside image")
    shape = [(pokeX * 8 - pokeWidth * 8 // 2, pokeY * 8 - pokeHeight * 8 // 2), (pokeX * 8 + pokeWidth * 8 // 2, pokeY * 8 + pokeHeight * 8 // 2)] 

    blank = Image.new("RGB", (imageWidth, imageHeight))

    if os.path.exists("outputs/original.png"):
        oImg = Image.open("outputs/original.png")
        pImg = Image.open("outputs/poked.png")
    else:
        oImg = blank
        pImg = blank

    oRec = ImageDraw.Draw(oImg) 
    pRec = ImageDraw.Draw(pImg)

    oRec.rectangle(shape, outline ="white")     
    pRec.rectangle(shape, outline ="white")
    
    return oImg, pImg

def display_poke_images(prompt, seed, num_inference_steps, poke=False, pokeX=None, pokeY=None, pokeHeight=None, pokeWidth=None, intermediate=False, progress=gr.Progress()):
    text_embeddings = get_text_embeddings(prompt)
    latents, modified_latents = generate_modified_latents(poke, seed, pokeX, pokeY, pokeHeight, pokeWidth)

    progress(0)
    images = generate_images(latents, text_embeddings, num_inference_steps, intermediate=intermediate)
    
    if not intermediate:
        images.save("outputs/original.png")

    if poke:
        progress(0.5)
        modImages = generate_images(modified_latents, text_embeddings, num_inference_steps, intermediate=intermediate)
        
        if not intermediate:
            modImages.save("outputs/poked.png")
    else:    
        modImages = None
    
    return images, modImages

__all__ = [
    "display_poke_images", 
    "visualize_poke"
]