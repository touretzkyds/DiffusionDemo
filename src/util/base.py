import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from src.util.params import *
from src.util.clip_config import *

def get_text_embeddings(prompt, tokenizer=tokenizer, text_encoder=text_encoder, torch_device=torch_device, batch_size=1):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    return text_embeddings

def generate_latents(seed, height=imageHeight, width=imageWidth, torch_device=torch_device, unet=unet, batch_size=1):
    generator = torch.Generator().manual_seed(int(seed))
    
    latents = torch.randn(
      (batch_size, unet.config.in_channels, height // 8, width // 8),
      generator=generator,
    ).to(torch_device)

    return latents

def generate_modified_latents(poke, seed, pokeX=None, pokeY=None, pokeHeight=None, pokeWidth=None, imageHeight=imageHeight, imageWidth=imageWidth):
    original_latents = generate_latents(seed, height=imageHeight, width=imageWidth)
    if poke:
        poke_latents = generate_latents(seed, height=pokeHeight, width=pokeWidth)

        x_origin = pokeX // 8 - poke_latents.shape[2] // 2          
        y_origin = pokeY // 8 - poke_latents.shape[3] // 2

        modified_latents = original_latents.clone()
        modified_latents[:,:,x_origin:x_origin+poke_latents.shape[2],y_origin:y_origin+poke_latents.shape[3]] = poke_latents
    else:
        modified_latents = None

    return original_latents, modified_latents

def convert_to_pil_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

def generate_images(latents, text_embeddings, num_inference_steps, unet=unet, guidance_scale=guidance_scale, vae=vae, scheduler=scheduler, intermediate=False):
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    images = []
    
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        if intermediate:
            Latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(Latents).sample
            images.append(convert_to_pil_image(image))

    if not intermediate:
        Latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(Latents).sample
        images = convert_to_pil_image(image)

    return images

def get_word_embeddings(prompt, tokenizer=tokenizer, text_encoder=text_encoder, torch_device=torch_device):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(torch_device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0].reshape(1,-1)   
    
    text_embeddings = text_embeddings.cpu().numpy()
    return text_embeddings/np.linalg.norm(text_embeddings)

def get_concat_embeddings(names):
    embeddings = []

    for name in names:
        embedding = get_word_embeddings(name)
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)
    return embeddings

def get_axis_embeddings(A, B):
    emb = []

    for a,b in zip(A,B):
        e = get_word_embeddings(a) - get_word_embeddings(b)
        emb.append(e)

    emb = np.vstack(emb)
    ax = np.average(emb, axis=0).reshape(1,-1)

    return ax

def calculate_residual(axis, axis_names, from_words=None, to_words=None):
    if axis_names[0] in axis_combinations:
        xembeddings = get_concat_embeddings(axis_combinations[axis_names[0]])
    else:
        xembeddings = get_concat_embeddings(from_words + to_words)

    if axis_names[2] in axis_combinations:
        zembeddings = get_concat_embeddings(axis_combinations[axis_names[2]])
    else:
        zembeddings = get_concat_embeddings(from_words + to_words)

    xprojections = xembeddings @ axis[0].T
    zprojections = zembeddings @ axis[2].T

    partial_residual = xembeddings - (xprojections.reshape(-1,1)*xembeddings)
    residual = partial_residual - (zprojections.reshape(-1,1)*zembeddings)

    residual = np.average(residual, axis=0).reshape(1,-1)
    residual = residual/np.linalg.norm(residual)

    return residual

__all__ = [
    "get_text_embeddings", 
    "generate_latents", 
    "generate_modified_latents", 
    "generate_images", 
    "get_word_embeddings", 
    "get_concat_embeddings", 
    "get_axis_embeddings", 
    "calculate_residual"
]  