import base64
import gradio as gr
from PIL import Image
from src.util import *
from io import BytesIO
from src.pipelines import *
from threading import Thread
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
    ]

    emb_children = [
        html.Img(
            src=generate_word_emb_vis(hover_data["text"]),
            style={"width": "100%", "height": "25px"},
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
    img.save(f"{txt}.png")
    return dcc.send_file(f"{txt}.png")

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

    with gr.Tab("Denoising"):
        gr.Markdown("Observe the intermediate images during denoising.")
        with gr.Row():
            with gr.Column():
                prompt_denoise = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_denoise = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps")
                
                with gr.Row():
                    seed_denoise = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_denoise = gr.Plot(value=generate_seed_vis(14), label="Seed")

                generate_images_button_denoise = gr.Button("Generate Images")
            
            with gr.Column():
                images_output_denoise = gr.Gallery(label="Images", selected_index=0)
                zip_output_denoise = gr.File(label="Download ZIP")
    
    @generate_images_button_denoise.click(inputs=[prompt_denoise, seed_denoise, num_inference_steps_denoise], outputs=[images_output_denoise, zip_output_denoise])
    def generate_images_wrapper(prompt, seed, num_inference_steps):
        images, _ = display_poke_images(prompt, seed, num_inference_steps, poke=False, intermediate=True)   
        fname = "denoising"
        tab_config = {
            "Tab"                           : "Denoising",
            "Prompt"                        : prompt, 
            "Number of Inference Steps"     : num_inference_steps,
            "Seed"                          : seed, 
        }
        export_as_zip(images, fname, tab_config)
        return images, f"outputs/{fname}.zip"
    
    seed_denoise.change(fn=generate_seed_vis, inputs=[seed_denoise], outputs=[seed_vis_denoise])
    
    with gr.Tab("Seeds"):
        gr.Markdown("Understand how different starting points in latent space can lead to different images.")
        with gr.Row():
            with gr.Column():
                prompt_seed = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_seed = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Seeds")
                num_inference_steps_seed = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                generate_images_button_seed = gr.Button("Generate Images")
            
            with gr.Column():
                images_output_seed = gr.Gallery(label="Images", selected_index=0)
                zip_output_seed = gr.File(label="Download ZIP")

    generate_images_button_seed.click(fn=display_seed_images, inputs=[prompt_seed, num_inference_steps_seed, num_images_seed], outputs=[images_output_seed, zip_output_seed])

    with gr.Tab("Perturbations"):
        gr.Markdown("Explore different perturbations from a point in latent space.")
        with gr.Row():
            with gr.Column():
                prompt_perturb = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_perturb = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Perturbations")
                perturbation_size_perturb = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Perturbation Size")
                num_inference_steps_perturb = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
                with gr.Row():
                    seed_perturb = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_perturb = gr.Plot(value=generate_seed_vis(14), label="Seed")

                generate_images_button_perturb = gr.Button("Generate Images")

            with gr.Column():
                images_output_perturb = gr.Gallery(label="Image", selected_index=0)  
                zip_output_perturb = gr.File(label="Download ZIP")  

    generate_images_button_perturb.click(fn=display_perturb_images, inputs=[prompt_perturb, seed_perturb, num_inference_steps_perturb, num_images_perturb, perturbation_size_perturb], outputs=[images_output_perturb, zip_output_perturb])
    seed_perturb.change(fn=generate_seed_vis, inputs=[seed_perturb], outputs=[seed_vis_perturb])

    with gr.Tab("Circular"):
        gr.Markdown("Generate a circular path in latent space and observe how the images vary along the path.")
        with gr.Row():
            with gr.Column():
                prompt_circular = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_circular = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Steps around the Circle")

                with gr.Row():
                    degree_circular = gr.Slider(minimum=0, maximum=360, step=1, value=360, label="Proportion of Circle", info="Enter the value in degrees")
                    step_size_circular = gr.Textbox(label="Step Size", value=360/5)

                num_inference_steps_circular = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
                with gr.Row():
                    seed_circular = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_circular = gr.Plot(value=generate_seed_vis(14), label="Seed")

                generate_images_button_circular = gr.Button("Generate Images")
           
            with gr.Column():
                images_output_circular = gr.Gallery(label="Image", selected_index=0)   
                gif_circular = gr.Image(label="GIF") 
                zip_output_circular = gr.File(label="Download ZIP")

    num_images_circular.change(fn=calculate_step_size, inputs=[num_images_circular, degree_circular], outputs=[step_size_circular])
    degree_circular.change(fn=calculate_step_size, inputs=[num_images_circular, degree_circular], outputs=[step_size_circular])
    generate_images_button_circular.click(fn=display_circular_images, inputs=[prompt_circular, seed_circular, num_inference_steps_circular, num_images_circular, degree_circular], outputs=[images_output_circular, gif_circular, zip_output_circular])
    seed_circular.change(fn=generate_seed_vis, inputs=[seed_circular], outputs=[seed_vis_circular])

    with gr.Tab("Interpolate"):
        gr.Markdown("Interpolate between the first and the second prompt, and observe how the output changes.")
        with gr.Row():
            with gr.Column():
                promptA = gr.Textbox(lines=1, label="First Prompt", value="Self-portrait oil painting, a beautiful man with golden hair, 8k")
                promptB = gr.Textbox(lines=1, label="Second Prompt", value="Self-portrait oil painting, a beautiful woman with golden hair, 8k")
                num_images_interpolate = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Interpolation Steps")
                num_inference_steps_interpolate = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
                with gr.Row():
                    seed_interpolate = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_interpolate = gr.Plot(value=generate_seed_vis(14), label="Seed")

                generate_images_button_interpolate = gr.Button("Generate Images")

            with gr.Column():
                images_output_interpolate = gr.Gallery(label="Interpolated Images", selected_index=0)
                gif_interpolate = gr.Image(label="GIF")
                zip_output_interpolate = gr.File(label="Download ZIP")

    generate_images_button_interpolate.click(fn=display_interpolate_images, inputs=[seed_interpolate, promptA, promptB, num_inference_steps_interpolate, num_images_interpolate], outputs=[images_output_interpolate, gif_interpolate, zip_output_interpolate])
    seed_interpolate.change(fn=generate_seed_vis, inputs=[seed_interpolate], outputs=[seed_vis_interpolate])

    with gr.Tab("Poke"):
        gr.Markdown("Perturb a region in the image and observe the effect.")
        with gr.Row():
            with gr.Column():
                prompt_poke = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_poke = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")

                with gr.Row():
                    seed_poke = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_poke = gr.Plot(value=generate_seed_vis(14), label="Seed")

                pokeX = gr.Slider(label="pokeX", minimum=0, maximum=64, step=1, value=32, info= "X coordinate of poke center")
                pokeY = gr.Slider(label="pokeY", minimum=0, maximum=64, step=1, value=32, info= "Y coordinate of poke center")
                pokeHeight = gr.Slider(label="pokeHeight", minimum=0, maximum=64, step=1, value=8, info= "Height of the poke")
                pokeWidth = gr.Slider(label="pokeWidth", minimum=0, maximum=64, step=1, value=8, info= "Width of the poke")
                
                generate_images_button_poke = gr.Button("Generate Images")   

            with gr.Column():
                original_images_output_poke = gr.Image(value=visualize_poke(32,32,8,8)[0], label="Original Image")
                poked_images_output_poke = gr.Image(value=visualize_poke(32,32,8,8)[1], label="Poked Image")
                zip_output_poke = gr.File(label="Download ZIP")

    pokeX.change(visualize_poke, inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=[original_images_output_poke, poked_images_output_poke])
    pokeY.change(visualize_poke, inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=[original_images_output_poke, poked_images_output_poke])
    pokeHeight.change(visualize_poke, inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=[original_images_output_poke, poked_images_output_poke])
    pokeWidth.change(visualize_poke, inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=[original_images_output_poke, poked_images_output_poke])
    seed_poke.change(fn=generate_seed_vis, inputs=[seed_poke], outputs=[seed_vis_poke])

    @generate_images_button_poke.click(inputs=[prompt_poke, seed_poke, num_inference_steps_poke, pokeX, pokeY, pokeHeight, pokeWidth], outputs=[original_images_output_poke, poked_images_output_poke, zip_output_poke])
    def generate_images_wrapper(prompt, seed, num_inference_steps, pokeX=pokeX, pokeY=pokeY, pokeHeight=pokeHeight, pokeWidth=pokeWidth):
        _, _ = display_poke_images(prompt, seed, num_inference_steps, poke=True, pokeX=pokeX, pokeY=pokeY, pokeHeight=pokeHeight, pokeWidth=pokeWidth, intermediate=False)
        images, modImages = visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth)
        fname = "poke"
        tab_config = {
            "Tab"                                   : "Poke",
            "Prompt"                                : prompt, 
            "Number of Inference Steps per Image"   : num_inference_steps,
            "Seed"                                  : seed,
            "PokeX"                                 : pokeX,
            "PokeY"                                 : pokeY,
            "PokeHeight"                            : pokeHeight,
            "PokeWidth"                             : pokeWidth,
        }
        imgs_list = []
        imgs_list.append((images, "Original Image"))
        imgs_list.append((modImages, "Poked Image"))

        export_as_zip(imgs_list, fname, tab_config)
        return images, modImages, f"outputs/{fname}.zip"
    
def run_dash():
    app.run(host="127.0.0.1", port="8000")

def run_gradio():
    demo.queue()
    demo.launch(share=True)

if __name__ == "__main__":
    thread = Thread(target=run_dash)
    thread.daemon = True
    thread.start()
    try:
        run_gradio()
    except KeyboardInterrupt:
        print("Server closed")