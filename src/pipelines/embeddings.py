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

age = get_axis_embeddings(young, old)
gender = get_axis_embeddings(masculine, feminine)
royalty = get_axis_embeddings(common, elite)

images = []
for example in examples:
    image = pipe(prompt=example, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    images.append("data:image/jpeg;base64, " + encoded_image)

axis = np.vstack([gender, royalty, age])
axis[1] = calculate_residual(axis, axis_names)

coords = get_concat_embeddings(examples) @ axis.T
coords[:, 1] = 5*(1.0 - coords[:, 1])

def update_fig():
    global coords, examples, fig
    fig.data[0].x = coords[:,0]
    fig.data[0].y = coords[:,1]
    fig.data[0].z = coords[:,2]
    fig.data[0].text = examples
    
    return (f'''
            <script>
                document.getElementById("html").src += "?rand={random.random()}"
            </script>
            <iframe id="html" src={dash_tunnel} style="width:100%; height:725px;"></iframe>
            ''')

def add_word(new_example):
    global coords, images, examples
    new_coord = get_concat_embeddings([new_example]) @ axis.T
    new_coord[:, 1] = 5*(1.0 - new_coord[:, 1])
    coords = np.vstack([coords, new_coord])

    image = pipe(prompt=new_example, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    images.append("data:image/jpeg;base64, " + encoded_image)
    examples.append(new_example)
    return update_fig()

def remove_word(new_example):
    global coords, images, examples
    examplesMap = { example: index for index, example in enumerate(examples) }
    index = examplesMap[new_example]
    
    coords = np.delete(coords, index, 0)
    images.pop(index)
    examples.pop(index)
    return update_fig()

def add_rem_word(new_examples):
    global examples
    new_examples = new_examples.replace(","," ").split()

    for new_example in new_examples:
        if new_example in examples:
            remove_word(new_example)
            gr.Info("Removed {}".format(new_example))
        else:
            tokens = tokenizer.encode(new_example)
            if len(tokens) != 3:
                gr.Warning(f"{new_example} not found in embeddings")
            else:
                add_word(new_example)
                gr.Info("Added {}".format(new_example))

    return update_fig()

def set_axis(axis_name, which_axis, from_words, to_words):
    global coords, examples, fig, axis_names

    if axis_name != "residual":
        from_words, to_words = from_words.replace(","," ").split(), to_words.replace(","," ").split()
        axis_emb = get_axis_embeddings(from_words, to_words)
        axis[axisMap[which_axis]] = axis_emb
        axis_names[axisMap[which_axis]] = axis_name

        for i, name in enumerate(axis_names):
            if name == "residual":
                axis[i] = calculate_residual(axis, axis_names, from_words, to_words, i)
                axis_names[i] = "residual"
    else:
        residual = calculate_residual(axis, axis_names, residual_axis=axisMap[which_axis])
        axis[axisMap[which_axis]] = residual
        axis_names[axisMap[which_axis]] = axis_name

    coords = get_concat_embeddings(examples) @ axis.T
    coords[:, 1] = 5*(1.0 - coords[:, 1])

    fig.update_layout(
        scene = dict(
            xaxis_title = axis_names[0],
            yaxis_title = axis_names[1],
            zaxis_title = axis_names[2],
        )
    )
    return update_fig()

def change_word(examples):
    examples = examples.replace(","," ").split()

    for example in examples:
        remove_word(example)
        add_word(example)
        gr.Info("Changed image for {}".format(example))
        
    return update_fig()

def clear_words():
    while examples:
        remove_word(examples[-1])
    return update_fig()

def generate_word_emb_vis(prompt):
    buf = BytesIO()
    emb = get_word_embeddings(prompt).reshape(77, 768)[1]
    plt.imsave(buf, [emb], cmap="inferno")
    img = "data:image/jpeg;base64, " + base64.b64encode(buf.getvalue()).decode('utf-8')
    return img

fig = px.scatter_3d(
                    x=coords[:,0], 
                    y=coords[:,1], 
                    z=coords[:,2], 
                    labels={
                            "x":axis_names[0],
                            "y":axis_names[1],
                            "z":axis_names[2],
                            },
                    text=examples,
                    height=750,
        )

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene_camera = dict(
        eye=dict(x=2, y=2, z=0.1)
    )
)

fig.update_traces(hoverinfo="none", hovertemplate=None)

__all__ = [
    "fig",
    "update_fig",
    "coords",
    "images",
    "examples",
    "add_word",
    "remove_word", 
    "add_rem_word", 
    "change_word", 
    "clear_words", 
    "generate_word_emb_vis",
    "set_axis",
    "axis"
]