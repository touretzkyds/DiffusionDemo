import os
import torch
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler
from src.main.utils import get_text_embeddings, generate_latents, generate_images           # comment this line if you want to run this file independently
# from utils import get_text_embeddings, generate_latents, generate_images                  # uncomment this line if you want to run this file independently

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def interpolate_prompts(promptA, promptB, num_interpolation_steps, tokenizer, text_encoder, torch_device):
    text_embeddingsA = get_text_embeddings(promptA, tokenizer, text_encoder, torch_device)
    text_embeddingsB = get_text_embeddings(promptB, tokenizer, text_encoder, torch_device)

    interpolated_embeddings = []
    
    for i in range(num_interpolation_steps):
        alpha = i / num_interpolation_steps
        interpolated_embedding = torch.lerp(text_embeddingsA, text_embeddingsB, alpha)
        interpolated_embeddings.append(interpolated_embedding)

    return interpolated_embeddings

def display_prompt_images(seed, promptA, promptB, num_inference_steps, num_images, imageHeight, imageWidth, guidance_scale, tokenizer, text_encoder, scheduler, unet, vae, torch_device, intermediate=False):
    latents = generate_latents(seed, imageHeight, imageWidth, torch_device, unet)
    text_embeddings = interpolate_prompts(promptA, promptB, num_images, tokenizer, text_encoder, torch_device)
    images = []
    for i in range(num_images):   
        image = generate_images(latents, text_embeddings[i], num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate)
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

    promptA = "a car driving on a highway in a city"
    promptB = "a truck driving on a highway in a city"

    num_images = 5                      # Number of images to generate
    num_inference_steps = 20            # 20+ for good results
    seed = 69420                        # Any integer

    images = display_prompt_images(seed, promptA, promptB, num_inference_steps, num_images, imageHeight, imageWidth, guidance_scale, tokenizer, text_encoder, scheduler, unet, vae, torch_device, intermediate=False)

    fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
    fig.suptitle("Interpolated Images")

    for i in range(num_images):
        ax[i].imshow(images[i][0])
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    ax[0].set_title("Prompt A")
    ax[num_images-1].set_title("Prompt B") 

    os.makedirs(f"./research/production/src/output/prompt_interpolation", exist_ok=True)
    plt.savefig(f"./research/production/src/output/prompt_interpolation/promptA_promptB.png")
