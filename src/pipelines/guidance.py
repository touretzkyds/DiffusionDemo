import gradio as gr
from src.util.base import *
from src.util.params import *


def display_guidance_images(
    prompt, seed, num_inference_steps, guidance_values, progress=gr.Progress()
):
    text_embeddings = get_text_embeddings(prompt)
    latents = generate_latents(seed)

    progress(0)
    images = []
    guidance_values = guidance_values.replace(",", " ").split()
    num_images = len(guidance_values)

    for i in range(num_images):
        progress(i / num_images)
        image = generate_images(
            latents,
            text_embeddings,
            num_inference_steps,
            guidance_scale=int(guidance_values[i]),
        )
        images.append((image, "{}".format(int(guidance_values[i]))))

    fname = "guidance"
    tab_config = {
        "Tab": "Guidance",
        "Prompt": prompt,
        "Guidance Scale Values": guidance_values,
        "Number of Inference Steps per Image": num_inference_steps,
        "Seed": seed,
    }
    export_as_zip(images, fname, tab_config)
    return images, f"outputs/{fname}.zip"


__all__ = ["display_guidance_images"]
