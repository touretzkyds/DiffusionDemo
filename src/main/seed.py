import matplotlib.pyplot as plt

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from utils import get_text_embeddings, generate_latents, generate_images 

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def display_seed_images(prompt, num_inference_steps, num_images, imageHeight, imageWidth, guidance_scale, tokenizer, text_encoder, unet, scheduler, vae, torch_device):
    text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder, torch_device)

    images = []
    for i in range(num_images):   
        latents = generate_latents(i, imageHeight, imageWidth, torch_device, unet)
        image = generate_images(latents, text_embeddings, num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate=False)
        images.append((image,i+1))

    return images   

if __name__ == "__main__":
    
    HF_ACCESS_TOKEN = "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"                      # Add your HuggingFace access token

    model_path = "segmind/tiny-sd"                                                 # Huggingface model path
    imageHeight, imageWidth = 512, 512                                             # Image size
    guidance_scale = 8                                                             # Guidance scale

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)
    scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

    prompt = "red balloon in the sky"
    num_images = 5                      # Number of images to generate
    num_inference_steps = 5             # 20+ for good results
    seed = 69420                        # Any integer

    images = display_seed_images(prompt, num_inference_steps, num_images, imageHeight, imageWidth, guidance_scale, tokenizer, text_encoder, unet, scheduler, vae, torch_device)

    fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
    fig.suptitle("Multiple Seeds")

    for i in range(num_images):
        ax[i].imshow(images[i][0])
        ax[i].set_title(images[i][1])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()