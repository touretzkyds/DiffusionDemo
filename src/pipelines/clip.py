import gradio as gr
import random
from threading import Thread

import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
from diffusers import LCMScheduler, AutoPipelineForText2Image

import base64
from io import BytesIO

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.express as px

from src.main.dims import masculine, feminine, young, old, common, elite, singular, plural, examples, axis_names, axis_combinations, axisMap

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "segmind/tiny-sd"
adapter_id = "akameswa/lcm-lora-tiny-sd"

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(torch_device)

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

age = get_axis_embeddings(young, old)
gender = get_axis_embeddings(masculine, feminine)
royalty = get_axis_embeddings(common, elite)

pipe = AutoPipelineForText2Image.from_pretrained(model_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

PeftModel.from_pretrained(pipe.unet, adapter_id)

images = []
for example in examples:
    image = pipe(prompt=example, num_inference_steps=4, guidance_scale=1.0).images[0]
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
            <iframe id="html" src="http://127.0.0.1:8000" style="width:100%; height:750px;"></iframe>
            ''')

def add_word(new_example):
    global coords, images, examples, fig
    new_coord = get_concat_embeddings([new_example]) @ axis.T
    new_coord[:, 1] = 5*(1.0 - new_coord[:, 1])

    coords = np.vstack([coords, new_coord])

    image = pipe(prompt=new_example, num_inference_steps=4, guidance_scale=1.0).images[0]
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    images.append("data:image/jpeg;base64, " + encoded_image)

    examples.append(new_example)
    return update_fig()

def remove_word(new_example):
    global coords, images, examples, fig
    examplesMap = { example: index for index, example in enumerate(examples) }
    index = examplesMap[new_example]
    
    coords = np.delete(coords, index, 0)
    images.pop(index)
    examples.pop(index)
    
    return update_fig()

def add_rem_word(new_example):
    global examples
    if new_example in examples:
        return remove_word(new_example)
    else:
        return add_word(new_example)

def set_axis(axis_name, which_axis, from_words, to_words):
    global coords, examples, fig, axis_names

    if axis_name != "residual":
        from_words, to_words = from_words.split(), to_words.split()
        axis_emb = get_axis_embeddings(from_words, to_words)
        axis[axisMap[which_axis]] = axis_emb
        axis_names[axisMap[which_axis]] = axis_name

        for i, name in enumerate(axis_names):
            if name == "residual":
                axis[i] = calculate_residual(axis, axis_names, from_words, to_words)
                axis_names[i] = "residual"
    else:
        residual = calculate_residual(axis, axis_names, from_words, to_words)
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

def change_word(example):
    remove_word(example)
    add_word(example)
    return update_fig()

def clear_words():
    while examples:
        remove_word(examples[-1])
    return update_fig()

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

app = Dash(__name__)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="tooltip"),
    ],
)

@callback(
    Output("tooltip", "show"),
    Output("tooltip", "bbox"),
    Output("tooltip", "children"),
    Output("tooltip", "direction"),

    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    direction = "left"
    index = hover_data['pointNumber']
    
    children = [
        html.Img(
            src=images[index],
            style={"width": "150px"},
        ),
    ]

    return True, bbox, children, direction


with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Demo")
    with gr.Tab("CLIP"):
        with gr.Row():
            output = gr.HTML(f'''
                    <iframe id="html" src="http://127.0.0.1:8000" style="width:100%; height:750px;"></iframe>
                    ''')
        with gr.Row():
            clear_words_button = gr.Button(value="Clear words")
        with gr.Row():
            word2add_rem = gr.Textbox(lines=2, label="Add/Remove word")
            word2change = gr.Textbox(lines=2, label="Change image for word")
        with gr.Row():
            add_rem_word_button = gr.Button(value="Add/Remove")
            change_word_button = gr.Button(value="Change")
        with gr.Accordion("Custom Semantic Dimensions", open=False):
            with gr.Accordion("Built-In Dimension 1"):
                with gr.Row():
                    axis_name_1 = gr.Textbox(label="Axis name", value="gender")
                    which_axis_1 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], value="X - Axis", label="Axis direction")
                with gr.Row():
                    set_axis_button_1 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_1 = gr.Textbox(lines=10, label="", value=f"""prince
husband
father
son
uncle""")
                        to_words_1 = gr.Textbox(lines=10, label="", value=f"""princess
wife
mother
daughter
aunt""")
            with gr.Accordion("Built-In Dimension 2"):
                with gr.Row():
                    axis_name_2 = gr.Textbox(label="Axis name", value="number")
                    which_axis_2 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], value="Z - Axis", label="Axis direction")
                with gr.Row():
                    set_axis_button_2 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_2 = gr.Textbox(lines=10, label="", value=f"""boys
girls
cats
puppies
computers""")
                        to_words_2 = gr.Textbox(lines=10, label="", value=f"""boy
girl
cat
puppy
computer""")
            with gr.Accordion("Built-In Dimension 3"):
                with gr.Row():
                    axis_name_3 = gr.Textbox(label="Axis name", value="age")
                    which_axis_3 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_3 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_3 = gr.Textbox(lines=10, label="", value=f"""man
woman
king
queen
father""")
                        to_words_3 = gr.Textbox(lines=10, label="", value=f"""boy
girl
prince
princess
son""")
            with gr.Accordion("Built-In Dimension 4"):
                with gr.Row():
                    axis_name_4 = gr.Textbox(label="Axis name", value="royalty")
                    which_axis_4 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_4 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_4 = gr.Textbox(lines=10, label="", value=f"""king
queen
prince
princess
woman""")
                        to_words_4 = gr.Textbox(lines=10, label="", value=f"""man
woman
boy
girl
duchess""")
            with gr.Accordion("Built-In Dimension 5"):
                with gr.Row():
                    axis_name_5 = gr.Textbox(label="Axis name", value="")
                    which_axis_5 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_5 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_5 = gr.Textbox(lines=10, label="")
                        to_words_5 = gr.Textbox(lines=10, label="")

    
    add_rem_word_button.click(fn=add_rem_word, inputs=[word2add_rem], outputs=[output])
    change_word_button.click(fn=change_word, inputs=[word2change], outputs=[output])
    clear_words_button.click(fn=clear_words, outputs=[output])
    
    set_axis_button_1.click(fn=set_axis, inputs=[axis_name_1, which_axis_1, from_words_1, to_words_1], outputs=[output])
    set_axis_button_2.click(fn=set_axis, inputs=[axis_name_2, which_axis_2, from_words_2, to_words_2], outputs=[output])
    set_axis_button_3.click(fn=set_axis, inputs=[axis_name_3, which_axis_3, from_words_3, to_words_3], outputs=[output])
    set_axis_button_4.click(fn=set_axis, inputs=[axis_name_4, which_axis_4, from_words_4, to_words_4], outputs=[output])
    set_axis_button_5.click(fn=set_axis, inputs=[axis_name_5, which_axis_5, from_words_5, to_words_5], outputs=[output])

def run_dash():
    app.run(host="127.0.0.1", port="8000")

def run_gradio():
    demo.queue()
    demo.launch(share=True)

if __name__ == "__main__":
    Thread(target=run_dash).start()
    run_gradio()
    