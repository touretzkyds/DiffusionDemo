import gradio as gr
from threading import Thread

from src.util.base import *
from src.util.params import *
from src.util.clip_config import *
from src.pipelines.clip import *
from dash import Dash, dcc, html, Input, Output, no_update, callback

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
    demo.launch()

if __name__ == "__main__":
    Thread(target=run_dash).start()
    run_gradio()
