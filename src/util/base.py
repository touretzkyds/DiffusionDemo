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


def export_as_gif(images, filename, frames_per_second=2, reverse=False):
    imgs = [img[0] for img in images]

    if reverse:
        imgs += imgs[2:-1][::-1]

    imgs[0].save(
        f"outputs/{filename}",
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


def export_as_zip(images, fname, tab_config=None):

    if not os.path.exists(f"outputs/{fname}.zip"):
        os.makedirs("outputs", exist_ok=True)

    with zipfile.ZipFile(f"outputs/{fname}.zip", "w") as img_zip:

        if tab_config:
            with open("outputs/config.txt", "w") as f:
                for key, value in tab_config.items():
                    f.write(f"{key}: {value}\n")
                f.close()

            img_zip.write("outputs/config.txt", "config.txt")

        for idx, img in enumerate(images):
            buff = io.BytesIO()
            img[0].save(buff, format="PNG")
            buff = buff.getvalue()
            max_num = len(images)
            num_leading_zeros = len(str(max_num))
            img_name = f"{{:0{num_leading_zeros}}}.png"
            img_zip.writestr(img_name.format(idx + 1), buff)


def read_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


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
]
