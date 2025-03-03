import gradio as gr
from src.util.base import *
from src.util.params import *


def display_negative_images(
    prompt, seed, num_inference_steps, negative_prompt="", progress=gr.Progress(), request: gr.Request = None
):
    text_embeddings = get_text_embeddings(prompt)
    text_embeddings_neg = get_text_embeddings(prompt, negative_prompt=negative_prompt)

    latents = generate_latents(seed)

    progress(0)
    images = generate_images(latents, text_embeddings, num_inference_steps)

    progress(0.5)
    images_neg = generate_images(latents, text_embeddings_neg, num_inference_steps)

    fname = "negative"
    tab_config = {
        "Tab": "Negative",
        "Prompt": prompt,
        "Negative Prompt": negative_prompt,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }

    imgs_list = []
    imgs_list.append((images, "Without Negative Prompt"))
    imgs_list.append((images_neg, "With Negative Prompt"))
    zip_path = export_as_zip(imgs_list, fname, tab_config, request=request)
    return images, images_neg, zip_path


__all__ = ["display_negative_images"]
