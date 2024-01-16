import gradio as gr
from src.util import *
from src.pipelines import *
from threading import Thread
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
                    <iframe id="html" src={dash_tunnel} style="width:100%; height:750px;"></iframe>
                    ''')
        with gr.Row():
            clear_words_button = gr.Button(value="Clear words")
        with gr.Row():
            word2add_rem = gr.Textbox(lines=1, label="Add/Remove word")
            word2change = gr.Textbox(lines=1, label="Change image for word")
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
                        from_words_1 = gr.Textbox(lines=1, label="", value="prince husband father son uncle")
                        to_words_1 = gr.Textbox(lines=1, label="", value="princess wife mother daughter aunt")
            with gr.Accordion("Built-In Dimension 2"):
                with gr.Row():
                    axis_name_2 = gr.Textbox(label="Axis name", value="number")
                    which_axis_2 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], value="Z - Axis", label="Axis direction")
                with gr.Row():
                    set_axis_button_2 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_2 = gr.Textbox(lines=1, label="", value="boys girls cats puppies computers")
                        to_words_2 = gr.Textbox(lines=1, label="", value="boy girl cat puppy computer")
            with gr.Accordion("Built-In Dimension 3"):
                with gr.Row():
                    axis_name_3 = gr.Textbox(label="Axis name", value="age")
                    which_axis_3 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_3 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_3 = gr.Textbox(lines=1, label="", value="man woman king queen father")
                        to_words_3 = gr.Textbox(lines=1, label="", value="boy girl prince princess son")
            with gr.Accordion("Built-In Dimension 4"):
                with gr.Row():
                    axis_name_4 = gr.Textbox(label="Axis name", value="royalty")
                    which_axis_4 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_4 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_4 = gr.Textbox(lines=1, label="", value="king queen prince princess woman")
                        to_words_4 = gr.Textbox(lines=1, label="", value="man woman boy girl duchess")
            with gr.Accordion("Built-In Dimension 5"):
                with gr.Row():
                    axis_name_5 = gr.Textbox(label="Axis name", value="")
                    which_axis_5 = gr.Dropdown(choices=["X - Axis", "Y - Axis", "Z - Axis"], label="Axis direction")
                with gr.Row():
                    set_axis_button_5 = gr.Button(value="Submit")
                with gr.Accordion("Words", open=False):
                    with gr.Row():
                        from_words_5 = gr.Textbox(lines=1, label="")
                        to_words_5 = gr.Textbox(lines=1, label="")

    
    add_rem_word_button.click(fn=add_rem_word, inputs=[word2add_rem], outputs=[output])
    change_word_button.click(fn=change_word, inputs=[word2change], outputs=[output])
    clear_words_button.click(fn=clear_words, outputs=[output])
    
    set_axis_button_1.click(fn=set_axis, inputs=[axis_name_1, which_axis_1, from_words_1, to_words_1], outputs=[output])
    set_axis_button_2.click(fn=set_axis, inputs=[axis_name_2, which_axis_2, from_words_2, to_words_2], outputs=[output])
    set_axis_button_3.click(fn=set_axis, inputs=[axis_name_3, which_axis_3, from_words_3, to_words_3], outputs=[output])
    set_axis_button_4.click(fn=set_axis, inputs=[axis_name_4, which_axis_4, from_words_4, to_words_4], outputs=[output])
    set_axis_button_5.click(fn=set_axis, inputs=[axis_name_5, which_axis_5, from_words_5, to_words_5], outputs=[output])

    with gr.Tab("Animation"):
        with gr.Row():
            with gr.Column():
                prompt_anim = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_anim = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Inference Steps per Image")
                seed_anim = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_anim = gr.Button("Generate Images")
            
            with gr.Column():
                images_output_anim = gr.Gallery(label="Images", selected_index=0)
    
    @generate_images_button_anim.click(inputs=[prompt_anim, seed_anim, num_inference_steps_anim], outputs=[images_output_anim])
    def generate_images_wrapper(prompt, seed, num_inference_steps):
        images, _ = display_poke_images(prompt, seed, num_inference_steps, poke=False, intermediate=True)
        return images
    
    with gr.Tab("Seed"):
        with gr.Row():
            with gr.Column():
                prompt_seed = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_seed = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                num_inference_steps_seed = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                generate_images_button_seed = gr.Button("Generate Images")
            
            with gr.Column():
                images_output_seed = gr.Gallery(label="Images", selected_index=0)

    with gr.Tab("Spread"):
        with gr.Row():
            with gr.Column():
                prompt_spread = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_spread = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                differentiation_spread = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Differentiation", info="The higher the differentiation, the more different the images will be")
                num_inference_steps_spread = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                seed_spread = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_spread = gr.Button("Generate Images")

            with gr.Column():
                images_output_spread = gr.Gallery(label="Image", selected_index=0)    

    with gr.Tab("Circular"):
        with gr.Row():
            with gr.Column():
                prompt_circular = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_circular = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                differentiation_circular = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Differentiation", info="The higher the differentiation, the more different the images will be")
                num_inference_steps_circular = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                seed_circular = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_circular = gr.Button("Generate Images")

            with gr.Column():
                images_output_circular = gr.Gallery(label="Image", selected_index=0)    

    with gr.Tab("Interpolate"):
        with gr.Row():
            with gr.Column():
                promptA = gr.Textbox(lines=1, label="Prompt from", value="Self-portrait oil painting, a beautiful man with golden hair, 8k")
                promptB = gr.Textbox(lines=1, label="Prompt to", value="Self-portrait oil painting, a beautiful woman with golden hair, 8k")
                num_images_1 = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                num_inference_steps_1 = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                seed_1 = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_1 = gr.Button("Generate Images")

            with gr.Column():
                images_output_1 = gr.Gallery(label="Interpolated Images", selected_index=0)
    
    with gr.Tab("Poke"):
        with gr.Row():
            with gr.Column():
                pokeX = gr.Slider(label="pokeX", minimum=0, maximum=imageWidth, step=1, value=256, info= "X coordinate of poke center")
                pokeY = gr.Slider(label="pokeY", minimum=0, maximum=imageHeight, step=1, value=256, info= "Y coordinate of poke center")
                pokeHeight = gr.Slider(label="pokeHeight", minimum=0, maximum=imageHeight, step=8, value=128, info= "Height of the poke")
                pokeWidth = gr.Slider(label="pokeWidth", minimum=0, maximum=imageWidth, step=8, value=128, info= "Width of the poke")
                visualize_poke_button = gr.Button("Visualize Poke")

            with gr.Column():
                visualize_poke_output = gr.Image(label="Poke Visualization")
                    
        with gr.Row():
            with gr.Column():
                prompt_0 = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_0 = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Inference Steps per Image")
                seed_0 = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_0 = gr.Button("Generate Images")
            
            with gr.Column():
                original_images_output_0 = gr.Gallery(label="Original Images", selected_index=0)
                poked_images_output_0 = gr.Gallery(label="Poked Images", selected_index=0)
               
    generate_images_button_seed.click(fn=display_seed_images, inputs=[prompt_seed, num_inference_steps_seed, num_images_seed], outputs=[images_output_seed])
    generate_images_button_spread.click(fn=display_spread_images, inputs=[prompt_spread, seed_spread, num_inference_steps_spread, num_images_spread, differentiation_spread], outputs=images_output_spread)
    generate_images_button_circular.click(fn=display_circular_images, inputs=[prompt_circular, seed_circular, num_inference_steps_circular, num_images_circular, differentiation_circular], outputs=images_output_circular)
    visualize_poke_button.click(fn=visualize_poke, inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=visualize_poke_output)

    @generate_images_button_0.click(inputs=[prompt_0, seed_0, num_inference_steps_0, pokeX, pokeY, pokeHeight, pokeWidth, ], outputs=[original_images_output_0, poked_images_output_0])
    def generate_images_wrapper(prompt, seed, num_inference_steps, pokeX=pokeX, pokeY=pokeY, pokeHeight=pokeHeight, pokeWidth=pokeWidth):
        images, modImages = display_poke_images(prompt, seed, num_inference_steps, poke=True, pokeX=pokeX, pokeY=pokeY, pokeHeight=pokeHeight, pokeWidth=pokeWidth, intermediate=True)
        return images, modImages
    
    generate_images_button_1.click(fn=display_interpolate_images, inputs=[seed_1, promptA, promptB, num_inference_steps_1, num_images_1], outputs=images_output_1)
    
def run_dash():
    app.run(host="127.0.0.1", port="8000")

def run_gradio():
    demo.queue()
    demo.launch()

if __name__ == "__main__":
    Thread(target=run_dash).start()
    run_gradio()