import os
import torch
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler
from src.main.utils import get_text_embeddings, generate_latents, generate_images        # comment this line if you want to run this file independently
# from utils import get_text_embeddings, generate_latents, generate_images               # uncomment this line if you want to run this file independently

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def display_similar_images(prompt, seed, num_inference_steps, num_images, imageHeight, imageWidth, differentiation, guidance_scale, tokenizer, text_encoder, unet, scheduler, vae, torch_device):
    text_embeddings = get_text_embeddings(prompt, tokenizer, text_encoder, torch_device)

    latents_x = generate_latents(seed, imageHeight, imageWidth, torch_device, unet)
    latents_y = generate_latents(seed*180, imageHeight, imageWidth, torch_device, unet)

    scale_x = torch.cos(torch.linspace(0, 1, 10)*2*torch.pi*differentiation).to(torch_device)
    scale_y = torch.sin(torch.linspace(0, 1, 10)*2*torch.pi*differentiation).to(torch_device)

    noise_x = torch.tensordot(scale_x, latents_x, dims=0)
    noise_y = torch.tensordot(scale_y, latents_y, dims=0)

    noise = noise_x + noise_y
    batched_noise = torch.split(noise, num_images)

    images = []
    for i in range(num_images):   
        image = generate_images(batched_noise[0][i], text_embeddings, num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate=False)
        images.append((image,i+1))

    return images   

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

    prompt = "red balloon in the sky"
    num_images = 5                      # Number of images to generate
    differentiation = 0.1               # Differentiation factor
    num_inference_steps = 20            # 20+ for good results
    seed = 69420                        # Any integer

    images = display_similar_images(prompt, seed, num_inference_steps, num_images, imageHeight, imageWidth, differentiation, guidance_scale, tokenizer, text_encoder, unet, scheduler, vae, torch_device)

    fig, ax = plt.subplots(1, num_images, dpi=1000, figsize=(num_images, 2))
    fig.suptitle("Similar Images")

    for i in range(num_images):
        ax[i].imshow(images[i][0])
        ax[i].set_title(images[i][1])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    os.makedirs(f"./research/production/src/output/similar_images", exist_ok=True)
    plt.savefig(f"./research/production/src/output/similar_images/{prompt}.png")