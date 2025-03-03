import gradio as gr
from src.util.base import *
from src.util.params import *


def display_seed_images(
    prompt, num_inference_steps, num_images, progress=gr.Progress(), request: gr.Request = None
):
    text_embeddings = get_text_embeddings(prompt)

    images = []
    progress(0)

    for i in range(num_images):
        progress(i / num_images)
        latents = generate_latents(i)
        image = generate_images(latents, text_embeddings, num_inference_steps)
        images.append((image, "{}".format(i + 1)))

    fname = "seeds"
    tab_config = {
        "Tab": "Seeds",
        "Prompt": prompt,
        "Number of Seeds": num_images,
        "Number of Inference Steps per Image": num_inference_steps,
    }
    zip_path = export_as_zip(images, fname, tab_config, request=request)
    return images, zip_path


__all__ = ["display_seed_images"]
