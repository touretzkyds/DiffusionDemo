import base64
import gradio as gr
from PIL import Image
from io import BytesIO
from threading import Thread

from src.util.base import *
from src.util.params import *
from src.util.clip_config import *
from src.pipelines.embeddings import *
from dash import Dash, dcc, html, Input, Output, no_update, callback

app = Dash(__name__)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True, style={"height": "90vh"}),
        dcc.Tooltip(id="tooltip"),
        html.Div(id="word-emb-txt", style={"background-color": "white"}),
        html.Div(id="word-emb-vis"),
        html.Div([
            html.Button(id="btn-download-image", hidden=True),
            dcc.Download(id="download-image"),
        ]),
    ],
)

@callback(
    Output("tooltip", "show"),
    Output("tooltip", "bbox"),
    Output("tooltip", "children"),
    Output("tooltip", "direction"),
    Output("word-emb-txt", "children"),
    Output("word-emb-vis", "children"),

    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update, no_update, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    direction = "left"
    index = hover_data['pointNumber']
    
    children = [
        html.Img(
            src=images[index],
            style={"width": "250px"},
        ),
        html.P(
            hover_data["text"],
            style={
                "color": "black", 
                "font-size": "20px",
                "text-align": "center",
                "background-color": "white",
                "margin": "5px",
            },
        ),
    ]

    emb_children = [
        html.Img(
            src=generate_word_emb_vis(hover_data["text"]),
            style={
                "width": "100%", 
                "height": "25px"
            },
        ),
    ]

    return True, bbox, children, direction, hover_data["text"], emb_children

@callback(
    Output("download-image", "data"),
    Input("graph", "clickData"),
)
def download_image(clickData):
    
    if clickData is None:
        return no_update
    
    click_data = clickData["points"][0]
    index = click_data["pointNumber"]
    txt = click_data["text"]

    img_encoded = images[index]
    img_decoded = base64.b64decode(img_encoded.split(",")[1])
    img = Image.open(BytesIO(img_decoded))
    img.save(f"outputs/{txt}.png")
    return dcc.send_file(f"outputs/{txt}.png")


with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Demo")
    with gr.Tab("Embeddings"):
        gr.Markdown("Visualize text embedding space in 3D with input texts and output images based on the chosen axis.")
        with gr.Row():
            output = gr.HTML(f'''
                    <iframe id="html" src="{dash_tunnel}" style="width:100%; height:700px;"></iframe>
                    ''')
        with gr.Row():
            word2add_rem = gr.Textbox(lines=1, label="Add/Remove word")
            word2change = gr.Textbox(lines=1, label="Change image for word")
            clear_words_button = gr.Button(value="Clear words")

        with gr.Accordion("Custom Semantic Dimensions", open=False):
            with gr.Row():
                axis_name_1 = gr.Textbox(label="Axis name", value="gender")
                which_axis_1 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_1"], label="Axis direction")
                from_words_1 = gr.Textbox(lines=1, label="Positive", value="prince husband father son uncle")
                to_words_1 = gr.Textbox(lines=1, label="Negative", value="princess wife mother daughter aunt")
                submit_1 = gr.Button("Submit")

            with gr.Row():
                axis_name_2 = gr.Textbox(label="Axis name", value="age")
                which_axis_2 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_2"], label="Axis direction")
                from_words_2 = gr.Textbox(lines=1, label="Positive", value="man woman king queen father")
                to_words_2 = gr.Textbox(lines=1, label="Negative", value="boy girl prince princess son")
                submit_2 = gr.Button("Submit")

            with gr.Row():
                axis_name_3 = gr.Textbox(label="Axis name", value="residual")
                which_axis_3 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_3"], label="Axis direction")
                from_words_3 = gr.Textbox(lines=1, label="Positive")
                to_words_3 = gr.Textbox(lines=1, label="Negative")
                submit_3 = gr.Button("Submit")

            with gr.Row():
                axis_name_4 = gr.Textbox(label="Axis name", value="number")
                which_axis_4 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_4"], label="Axis direction")
                from_words_4 = gr.Textbox(lines=1, label="Positive", value="boys girls cats puppies computers")
                to_words_4 = gr.Textbox(lines=1, label="Negative", value="boy girl cat puppy computer")
                submit_4 = gr.Button("Submit")

            with gr.Row():
                axis_name_5 = gr.Textbox(label="Axis name", value="royalty")
                which_axis_5 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_5"], label="Axis direction")
                from_words_5 = gr.Textbox(lines=1, label="Positive", value="king queen prince princess duchess")
                to_words_5 = gr.Textbox(lines=1, label="Negative", value="man woman boy girl woman")
                submit_5 = gr.Button("Submit")

            with gr.Row():
                axis_name_6 = gr.Textbox(label="Axis name")
                which_axis_6 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis", "---"], value=whichAxisMap["which_axis_6"], label="Axis direction")
                from_words_6 = gr.Textbox(lines=1, label="Positive")
                to_words_6 = gr.Textbox(lines=1, label="Negative")
                submit_6 = gr.Button("Submit")

    
    @word2add_rem.submit(inputs=[word2add_rem], outputs=[output, word2add_rem])
    def add_rem_word_and_clear(words):
        return add_rem_word(words), ""

    @word2change.submit(inputs=[word2change], outputs=[output, word2change])
    def change_word_and_clear(word):
        return change_word(word), ""  

    clear_words_button.click(fn=clear_words, outputs=[output])

    @submit_1.click(inputs=[axis_name_1, which_axis_1, from_words_1, to_words_1], outputs=[output, which_axis_2, which_axis_3, which_axis_4, which_axis_5, which_axis_6])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):

        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_1"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_2"], whichAxisMap["which_axis_3"], whichAxisMap["which_axis_4"], whichAxisMap["which_axis_5"], whichAxisMap["which_axis_6"]
    
    @submit_2.click(inputs=[axis_name_2, which_axis_2, from_words_2, to_words_2], outputs=[output, which_axis_1, which_axis_3, which_axis_4, which_axis_5, which_axis_6])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):
            
        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_2"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_1"], whichAxisMap["which_axis_3"], whichAxisMap["which_axis_4"], whichAxisMap["which_axis_5"], whichAxisMap["which_axis_6"]

    @submit_3.click(inputs=[axis_name_3, which_axis_3, from_words_3, to_words_3], outputs=[output, which_axis_1, which_axis_2, which_axis_4, which_axis_5, which_axis_6])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):
            
        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_3"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_1"], whichAxisMap["which_axis_2"], whichAxisMap["which_axis_4"], whichAxisMap["which_axis_5"], whichAxisMap["which_axis_6"]

    @submit_4.click(inputs=[axis_name_4, which_axis_4, from_words_4, to_words_4], outputs=[output, which_axis_1, which_axis_2, which_axis_3, which_axis_5, which_axis_6])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):

        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_4"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_1"], whichAxisMap["which_axis_2"], whichAxisMap["which_axis_3"], whichAxisMap["which_axis_5"], whichAxisMap["which_axis_6"]

    @submit_5.click(inputs=[axis_name_5, which_axis_5, from_words_5, to_words_5], outputs=[output, which_axis_1, which_axis_2, which_axis_3, which_axis_4, which_axis_6])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):
            
        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_5"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_1"], whichAxisMap["which_axis_2"], whichAxisMap["which_axis_3"], whichAxisMap["which_axis_4"], whichAxisMap["which_axis_6"]
    
    @submit_6.click(inputs=[axis_name_6, which_axis_6, from_words_6, to_words_6], outputs=[output, which_axis_1, which_axis_2, which_axis_3, which_axis_4, which_axis_5])
    def set_axis_wrapper(axis_name, which_axis, from_words, to_words):
                
        for ax in whichAxisMap:
            if whichAxisMap[ax] == which_axis:
                whichAxisMap[ax] = "---"

        whichAxisMap["which_axis_6"] = which_axis
        return set_axis(axis_name, which_axis, from_words, to_words), whichAxisMap["which_axis_1"], whichAxisMap["which_axis_2"], whichAxisMap["which_axis_3"], whichAxisMap["which_axis_4"], whichAxisMap["which_axis_5"]

def run_dash():
    app.run(host="127.0.0.1", port="8000")

def run_gradio():
    demo.queue()
    demo.launch()

if __name__ == "__main__":
    thread = Thread(target=run_dash)
    thread.daemon = True
    thread.start()
    try:
        run_gradio()
    except KeyboardInterrupt:
        print("Server closed")
