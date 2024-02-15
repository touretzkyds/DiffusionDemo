import torch
import gradio as gr
from src.util.base import *
from src.util.params import *
from diffusers import AutoPipelineForInpainting

inpaint_pipe = AutoPipelineForInpainting.from_pretrained(inpaint_model_path).to(torch_device)

def inpaint(dict, num_inference_steps, seed, prompt="", progress=gr.Progress()):
    progress(0)
    mask = dict["layers"][0].convert("RGB").resize((imageHeight, imageWidth))
    init_image = dict["background"].convert("RGB").resize((imageHeight, imageWidth))
    output = inpaint_pipe(prompt = prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=torch.Generator().manual_seed(seed))
    progress(1)

    fname = "inpainting"
    tab_config = {
        "Tab"                                   : "Inpainting",
        "Prompt"                                : prompt, 
        "Number of Inference Steps per Image"   : num_inference_steps,
        "Seed"                                  : seed,
    }

    imgs_list = []
    imgs_list.append((output.images[0], "Inpainted Image"))
    imgs_list.append((mask, "Mask"))

    export_as_zip(imgs_list, fname, tab_config)
    return output.images[0], f"outputs/{fname}.zip"

__all__ = [
    "inpaint"
]