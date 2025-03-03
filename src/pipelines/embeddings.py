import random
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO
import plotly.express as px

from src.util.base import *
from src.util.params import *
from src.util.clip_config import *
from src.util.session import session_manager

age = get_axis_embeddings(young, old)
gender = get_axis_embeddings(masculine, feminine)
royalty = get_axis_embeddings(common, elite)

def initialize_state():
    initial_images = []
    for example in examples:
        image = pipe(
            prompt=example,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        initial_images.append("data:image/jpeg;base64, " + encoded_image)

    initial_axis = np.vstack([gender, royalty, age])
    initial_axis[1] = calculate_residual(initial_axis, axis_names)

    initial_coords = get_concat_embeddings(examples) @ initial_axis.T
    initial_coords[:, 1] = 5 * (1.0 - initial_coords[:, 1])

    fig = px.scatter_3d(
        x=initial_coords[:, 0],
        y=initial_coords[:, 1],
        z=initial_coords[:, 2],
        labels={
            "x": axis_names[0],
            "y": axis_names[1],
            "z": axis_names[2],
        },
        text=examples,
        height=750,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0), 
        scene_camera=dict(eye=dict(x=2, y=2, z=0.1))
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return initial_axis, initial_coords, initial_images, fig

def update_fig(coords, examples, fig):
    fig.data[0].x = coords[:, 0]
    fig.data[0].y = coords[:, 1]
    fig.data[0].z = coords[:, 2]
    fig.data[0].text = examples

    return f"""
            <script>
                document.getElementById("html").src += "?rand={random.random()}"
            </script>
            <iframe id="html" src={dash_tunnel} style="width:100%; height:725px;"></iframe>
            """

def add_word(new_example, coords_state, images_state, examples_state, axis_state, request: gr.Request = None):
    coords = coords_state.copy()
    images = images_state.copy()
    examples = examples_state.copy()
    axis = axis_state.copy()

    new_coord = get_concat_embeddings([new_example]) @ axis.T
    new_coord[:, 1] = 5 * (1.0 - new_coord[:, 1])
    coords = np.vstack([coords, new_coord])

    image = pipe(
        prompt=new_example,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    if request:
        session_dir = session_manager.get_session_path(request.session_hash)
        image_path = session_dir / f"{new_example}.png"
        image.save(image_path)
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
    else:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    images.append("data:image/jpeg;base64, " + encoded_image)
    examples.append(new_example)
    
    return coords, images, examples

def remove_word(new_example, coords_state, images_state, examples_state, request: gr.Request = None):
    coords = coords_state.copy()
    images = images_state.copy()
    examples = examples_state.copy()

    examplesMap = {example: index for index, example in enumerate(examples)}
    index = examplesMap[new_example]

    coords = np.delete(coords, index, 0)
    images.pop(index)
    examples.pop(index)
    
    if request:
        session_dir = session_manager.get_session_path(request.session_hash)
        image_path = session_dir / f"{new_example}.png"
        if image_path.exists():
            image_path.unlink()
            
    return coords, images, examples

def add_rem_word(new_examples, coords_state, images_state, examples_state, axis_state, request: gr.Request = None):
    coords = coords_state.copy()
    images = images_state.copy()
    examples = examples_state.copy()
    axis = axis_state.copy()

    new_examples = new_examples.replace(",", " ").split()

    for new_example in new_examples:
        if new_example in examples:
            coords, images, examples = remove_word(new_example, coords, images, examples, request=request)
            gr.Info("Removed {}".format(new_example))
        else:
            tokens = tokenizer.encode(new_example)
            if len(tokens) != 3:
                gr.Warning(f"{new_example} not found in embeddings")
            else:
                coords, images, examples = add_word(new_example, coords, images, examples, axis, request=request)
                gr.Info("Added {}".format(new_example))

    return coords, images, examples

def set_axis(axis_name, which_axis, from_words, to_words, coords_state, examples_state, axis_state, axis_names_state):
    coords = coords_state.copy()
    examples = examples_state.copy()
    axis = axis_state.copy()
    axis_names = axis_names_state.copy()

    if axis_name != "residual":
        from_words, to_words = (
            from_words.replace(",", " ").split(),
            to_words.replace(",", " ").split(),
        )
        axis_emb = get_axis_embeddings(from_words, to_words)
        axis[axisMap[which_axis]] = axis_emb
        axis_names[axisMap[which_axis]] = axis_name

        for i, name in enumerate(axis_names):
            if name == "residual":
                axis[i] = calculate_residual(axis, axis_names, from_words, to_words, i)
                axis_names[i] = "residual"
    else:
        residual = calculate_residual(
            axis, axis_names, residual_axis=axisMap[which_axis]
        )
        axis[axisMap[which_axis]] = residual
        axis_names[axisMap[which_axis]] = axis_name

    coords = get_concat_embeddings(examples) @ axis.T
    coords[:, 1] = 5 * (1.0 - coords[:, 1])

    return coords, axis, axis_names

def change_word(examples_str, coords_state, images_state, examples_state, axis_state, request: gr.Request = None):
    coords = coords_state.copy()
    images = images_state.copy()
    examples = examples_state.copy()
    axis = axis_state.copy()

    examples_list = examples_str.replace(",", " ").split()

    for example in examples_list:
        coords, images, examples = remove_word(example, coords, images, examples, request=request)
        coords, images, examples = add_word(example, coords, images, examples, axis, request=request)
        gr.Info("Changed image for {}".format(example))

    return coords, images, examples

def clear_words(coords_state, images_state, examples_state, request: gr.Request = None):
    coords = coords_state.copy()
    images = images_state.copy()
    examples = examples_state.copy()

    while examples:
        coords, images, examples = remove_word(examples[-1], coords, images, examples, request=request)
    return coords, images, examples

def generate_word_emb_vis(prompt, request: gr.Request = None):
    buf = BytesIO()
    emb = get_word_embeddings(prompt).reshape(77, 768)[1]
    plt.imsave(buf, [emb], cmap="inferno")
    img = "data:image/jpeg;base64, " + base64.b64encode(buf.getvalue()).decode("utf-8")
    return img

__all__ = [
    "initialize_state",
    "update_fig",
    "add_word",
    "remove_word",
    "add_rem_word",
    "change_word",
    "clear_words",
    "generate_word_emb_vis",
    "set_axis",
]
