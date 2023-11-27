import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler
from src.main.utils import get_text_embeddings, generate_modified_latents, generate_images        # comment this line if you want to run this file independently
# from utils import get_text_embeddings, generate_modified_latents, generate_images               # uncomment this line if you want to run this file independently

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, imageHeight, imageWidth):
    if pokeX - pokeWidth // 2 < 0 or pokeX + pokeWidth // 2 > imageWidth or pokeY - pokeHeight // 2 < 0 or pokeY + pokeHeight // 2 > imageHeight:
        print("Error: Poking region outside image")
    shape = [(pokeX - pokeWidth // 2, pokeY - pokeHeight // 2), (pokeX + pokeWidth // 2, pokeY + pokeHeight // 2)] 
    img = Image.new("RGB", (imageHeight, imageWidth))
    rec = ImageDraw.Draw(img) 
    rec.rectangle(shape, outline ="white") 
    return img


def display_images(prompt, pokeX, pokeY, heightPoke, widthPoke, seed, num_inference_steps, tokenizer, text_encoder, unet, scheduler, vae, guidance_scale, torch_device, imageHeight, imageWidth, intermediate=True):
    text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder, torch_device)
    latents, modified_latents = generate_modified_latents(pokeX, pokeY, seed, heightPoke, widthPoke, imageHeight, imageWidth, torch_device, unet)
    images = generate_images(latents, text_embeddings, num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate)
    modImages = generate_images(modified_latents, text_embeddings, num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate)
    
    return images, modImages

if __name__ == "__main__":

    HF_ACCESS_TOKEN = ""                                                           # Add your HuggingFace access token

    model_path = "/home/akameswa/research/models/tiny-sd"                          # Huggingface model path
    imageHeight, imageWidth = 512, 512                                             # Image size
    guidance_scale = 8                                                             # Guidance scale

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)
    scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

    prompt = "dog in car"
    num_inference_steps = 20            # 20+ for good results
    imageWidth, imageHeight = 512, 512  # Image size
    seed = 69420                        # Any integer

    intermediate = True                 # Set true if you want to see intermediate images
    pokeX, pokeY = 256, 256             # Poke coordinates
    pokeHeight, pokeWidth = 128, 128    # Poke size

    if pokeX - pokeWidth // 2 < 0 or pokeX + pokeWidth // 2 > imageWidth or pokeY - pokeHeight // 2 < 0 or pokeY + pokeHeight // 2 > imageHeight:
        print("Error: Poking region outside image")

    visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, imageHeight, imageWidth)

    images, modImages = display_images(prompt, pokeX, pokeY, pokeHeight, pokeWidth, seed, num_inference_steps, tokenizer, text_encoder, unet, scheduler, vae, guidance_scale, torch_device, imageHeight, imageWidth, intermediate=intermediate)

    if intermediate:
        fig, ax = plt.subplots(2, num_inference_steps, dpi=1000, figsize=(num_inference_steps, 3))
        fig.suptitle("Original (Top) vs Poked (Bottom)")

        for i in range(num_inference_steps):
            ax[0][i].set_title(f"Step {i+1}")
            ax[0][i].imshow(images[i])
            ax[0][i].set_xticks([])
            ax[0][i].set_yticks([])

            ax[1][i].set_title(f"Step {i+1}")
            ax[1][i].imshow(modImages[i])
            ax[1][i].set_xticks([])
            ax[1][i].set_yticks([])

    else:
        fig, ax = plt.subplots(2, 1, dpi=1000, figsize=(1, 2))
        fig.suptitle("Original (Top) vs Poked (Bottom)")
        ax[0].imshow(images)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(modImages)
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    plt.savefig(f"./research/production/src/output/poke/{prompt}.png")

