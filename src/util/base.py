import io
import os
import torch
import zipfile
import numpy as np
import gradio as gr
from PIL import Image
from tqdm.auto import tqdm
from src.util.params import *
from src.util.clip_config import *
import matplotlib.pyplot as plt
import json
from src.util.session import session_manager


def get_text_embeddings(
    prompt,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    torch_device=torch_device,
    batch_size=1,
    negative_prompt="",
):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [negative_prompt] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


def generate_latents(
    seed,
    height=imageHeight,
    width=imageWidth,
    torch_device=torch_device,
    unet=unet,
    batch_size=1,
):
    generator = torch.Generator().manual_seed(int(seed))

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(torch_device)

    return latents


def generate_modified_latents(
    poke,
    seed,
    pokeX=None,
    pokeY=None,
    pokeHeight=None,
    pokeWidth=None,
    imageHeight=imageHeight,
    imageWidth=imageWidth,
):
    original_latents = generate_latents(seed, height=imageHeight, width=imageWidth)
    if poke:
        np.random.seed(seed)
        poke_latents = generate_latents(
            np.random.randint(0, 100000), height=pokeHeight * 8, width=pokeWidth * 8
        )

        x_origin = pokeX - pokeWidth // 2
        y_origin = pokeY - pokeHeight // 2

        modified_latents = original_latents.clone()
        modified_latents[
            :, :, y_origin : y_origin + pokeHeight, x_origin : x_origin + pokeWidth
        ] = poke_latents
    else:
        modified_latents = None

    return original_latents, modified_latents


def convert_to_pil_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]


def generate_images(
    latents,
    text_embeddings,
    num_inference_steps,
    unet=unet,
    guidance_scale=guidance_scale,
    vae=vae,
    scheduler=scheduler,
    intermediate=False,
    progress=gr.Progress(),
):
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    images = []
    i = 1

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if intermediate:
            progress(((1000 - t) / 1000))
            Latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(Latents).sample
                images.append((convert_to_pil_image(image), "{}".format(i)))

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        i += 1

    if not intermediate:
        Latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(Latents).sample
        images = convert_to_pil_image(image)

    return images


def get_word_embeddings(
    prompt, tokenizer=tokenizer, text_encoder=text_encoder, torch_device=torch_device
):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(torch_device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0].reshape(1, -1)

    text_embeddings = text_embeddings.cpu().numpy()
    return text_embeddings / np.linalg.norm(text_embeddings)


def get_concat_embeddings(names, merge=False):
    embeddings = []

    for name in names:
        embedding = get_word_embeddings(name)
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)

    if merge:
        embeddings = np.average(embeddings, axis=0).reshape(1, -1)

    return embeddings


def get_axis_embeddings(A, B):
    emb = []

    for a, b in zip(A, B):
        e = get_word_embeddings(a) - get_word_embeddings(b)
        emb.append(e)

    emb = np.vstack(emb)
    ax = np.average(emb, axis=0).reshape(1, -1)

    return ax


def calculate_residual(
    axis, axis_names, from_words=None, to_words=None, residual_axis=1
):
    axis_indices = [0, 1, 2]
    axis_indices.remove(residual_axis)

    if axis_names[axis_indices[0]] in axis_combinations:
        fembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[0]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[0]]] = from_words + to_words
        fembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    if axis_names[axis_indices[1]] in axis_combinations:
        sembeddings = get_concat_embeddings(
            axis_combinations[axis_names[axis_indices[1]]], merge=True
        )
    else:
        axis_combinations[axis_names[axis_indices[1]]] = from_words + to_words
        sembeddings = get_concat_embeddings(from_words + to_words, merge=True)

    fprojections = fembeddings @ axis[axis_indices[0]].T
    sprojections = sembeddings @ axis[axis_indices[1]].T

    partial_residual = fembeddings - (fprojections.reshape(-1, 1) * fembeddings)
    residual = partial_residual - (sprojections.reshape(-1, 1) * sembeddings)

    return residual


def calculate_step_size(num_images, start_degree_circular, end_degree_circular):
    return (end_degree_circular - start_degree_circular) / (num_images)


def generate_seed_vis(seed):
    np.random.seed(seed)
    emb = np.random.rand(15)
    plt.close()
    plt.switch_backend("agg")
    plt.figure(figsize=(10, 0.5))
    plt.imshow([emb], cmap="viridis")
    plt.axis("off")
    return plt


def export_as_gif(images, filename="output.gif", duration=500, reverse=False, request: gr.Request = None):
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    gif_path = session_dir / filename
    
    pil_images = []
    if isinstance(images, list) and isinstance(images[0], tuple):
        pil_images = [img for img, _ in images]
    else:
        pil_images = images
    
    if reverse:
        pil_images = pil_images + pil_images[::-1]
    
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    return str(gif_path)


def export_as_zip(images, fname, tab_config, request: gr.Request = None):
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    
    if isinstance(images, list):
        for i, (image, caption) in enumerate(images):
            image.save(session_dir / f"{fname}_{caption}.png")
    else:
        images.save(session_dir / f"{fname}.png")
    
    with open(session_dir / f"{fname}_config.txt", "w") as f:
        json.dump(tab_config, f, indent=4)
    
    zip_path = session_dir / f"{fname}.zip"
    os.system(f"cd {session_dir} && zip {fname}.zip {fname}*.png {fname}_config.txt")
    return str(zip_path)


def read_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def get_user_dir(session_hash):
    """Get the main directory for a specific user's session"""
    if not session_hash:
        return None
    user_dir = session_manager.get_session_path(session_hash)
    print(f"User directory path: {user_dir.absolute()}")
    return user_dir


def get_user_examples_dir(session_hash):
    """Get the examples directory for a specific user's session"""
    if not session_hash:
        return None
    examples_dir = session_manager.get_file_path(session_hash, "examples")
    examples_dir.mkdir(exist_ok=True)
    return examples_dir


def get_user_viz_dir(session_hash):
    """Get the visualizations directory for a specific user's session"""
    if not session_hash:
        return None
    viz_dir = session_manager.get_file_path(session_hash, "visualizations")
    viz_dir.mkdir(exist_ok=True)
    return viz_dir

__all__ = [
    "get_text_embeddings",
    "generate_latents",
    "generate_modified_latents",
    "generate_images",
    "get_word_embeddings",
    "get_concat_embeddings",
    "get_axis_embeddings",
    "calculate_residual",
    "calculate_step_size",
    "generate_seed_vis",
    "export_as_gif",
    "export_as_zip",
    "read_html",
    "get_user_dir",
    "get_user_examples_dir",
    "get_user_viz_dir",
]
