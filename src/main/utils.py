import torch
from PIL import Image
from tqdm.auto import tqdm

def get_text_embeddings(prompt, tokenizer, text_encoder, torch_device, batch_size=1):
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

def generate_latents(seed, height, width, torch_device, unet, batch_size=1):
    generator = torch.Generator().manual_seed(int(seed))
    
    latents = torch.randn(
      (batch_size, unet.config.in_channels, height // 8, width // 8),
      generator=generator,
    ).to(torch_device)

    return latents

def generate_modified_latents(pokeX, pokeY, seed, pokeHeight, pokeWidth, imageHeight, imageWidth, torch_device, unet):
    original_latents = generate_latents(seed, imageHeight, imageWidth, torch_device, unet)
    poke_latents = generate_latents(seed, pokeHeight, pokeWidth, torch_device, unet)

    x_origin = pokeX // 8 - poke_latents.shape[2] // 2          
    y_origin = pokeY // 8 - poke_latents.shape[3] // 2

    modified_latents = original_latents.clone()
    modified_latents[:,:,x_origin:x_origin+poke_latents.shape[2],y_origin:y_origin+poke_latents.shape[3]] = poke_latents

    return original_latents, modified_latents

def convert_to_pil_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

def generate_images(latents, text_embeddings, num_inference_steps, unet, guidance_scale, vae, scheduler, intermediate):
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
