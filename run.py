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
    img.save(f"{txt}.png")
    return dcc.send_file(f"{txt}.png")

with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Demo")
    with gr.Tab("Embeddings"):
        gr.Markdown("Visualize text embedding space in 3D with input texts and output images based on the chosen axis.")
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        **Basic Exploration**  
                        The top part of the embeddings tab is the 3D plot of semantic feature space. 
                        At the bottom of the tab there are expandable panels that can be opened to reveal more advanced features

                        **Explore the 3D semantic feature space:**  
                        Click and drag in the 3D semantic feature space to rotate the view. 
                        Use the scroll wheel to zoom in and out. Hold down the control key and click and drag to pan the view

                        **Find the generated image:**  
                        Hover over a point in the semantic feature space, and a window will pop up showing a generated image from this one-word prompt. 
                        On left click, the image will be downloaded.

                        **Find the embedding vector display:**  
                        Hover over a word in the 3D semantic feature space, and an embedding vector display at the bottom of the tab shows the corresponding embedding vector.

                        **Add/remove words from the 3D plot:**  
                        Type a word in the Add/Remove word text box below the 3D plot to add a word to the plot, or if the word is already present, remove it from the plot. 
                        You can also type multiple words separated by spaces or commas.

                        **Change image for word in the 3D plot:**  
                        Type a word in the Change image for word text box below the 3D plot to generate a new image for the corresponding word in the plot.
                        
                        **Semantic Dimensions**  
                        **Select a different semantic dimension.**  
                        Open the Custom Semantic Dimensions panel and choose another dimension for the X or Y or Z axis. 
                        See how the display changes.

                        **Alter a semantic dimension.**  
                        Examine the positive and negative word pairs used to define the semantic dimension. 
                        You can change these pairs to alter the semantic dimension. 

                        **Define a new semantic dimension.**  
                        Pick a new semantic dimension that you can define using pairs of opposed words. 
                        For example, you could define a "tense" dimension with pairs such as eat/ate, go/went, see/saw, and is/was to contrast present and past tense forms of verbs.
                        """)
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
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        This tab displays the intermediate images generated during the denoising process. 
                        Seeing these intermediate images provides insight into how the diffusion model progressively adds detail at each step.
                        """)
        with gr.Row():
            with gr.Column():
                prompt_denoise = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_denoise = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps")
                
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
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        Seeds create the initial noise that gets refined into the target image. 
                        Different seeds produce different noise patterns, hence the target image will differ even when prompted by the same text. 
                        This tab produces multiple target images from the same text prompt to showcase how changing the seed changes the target image.

                        **Additional Controls:**  
                        **Number of Seeds:**  
                        Specify how many seed values to use.
                        """)
        with gr.Row():
            with gr.Column():
                prompt_seed = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_seed = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Seeds")
                num_inference_steps_seed = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                generate_images_button_seed = gr.Button("Generate Images")
            
            with gr.Column():
                images_output_seed = gr.Gallery(label="Images", selected_index=0)
                zip_output_seed = gr.File(label="Download ZIP")

    generate_images_button_seed.click(fn=display_seed_images, inputs=[prompt_seed, num_inference_steps_seed, num_images_seed], outputs=[images_output_seed, zip_output_seed])

    with gr.Tab("Perturbations"):
        gr.Markdown("Explore different perturbations from a point in latent space.")
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        This tab enables the exploration of the latent space around a seed. 
                        Perturbing the noise from an initial seed towards the noise from a different seed illustrates the variations in images obtainable from a local region of latent space. 
                        Using a small perturbation size produces target images that closely resemble the one from the initial seed. 
                        Larger perturbations traverse more distance in latent space towards the second seed, resulting in greater variation in the generated images.

                        **Additional Controls:**  
                        **Number of Perturbations:**  
                        Specify the number of perturbations to create, i.e., the number of seeds to use. More perturbations produce more images.

                        **Perturbation Size:**  
                        Controls the perturbation magnitude, ranging from 0 to 1. 
                        With a value of 0, all images will match the one from the initial seed. 
                        With a value of 1, images will have no connection to the initial seed. 
                        A value such as 0.1 is recommended.
                        """)
        with gr.Row():
            with gr.Column():
                prompt_perturb = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_perturb = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Perturbations")
                perturbation_size_perturb = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Perturbation Size")
                num_inference_steps_perturb = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
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
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        This tab generates a circular trajectory through latent space that begins and ends with the same image. 
                        If we specify a large number of steps around the circle, the successive images will be closely related, resulting in a gradual deformation that produces a nice animation.

                        **Additional Controls:**  
                        **Number of Steps around the Circle:**  
                        Specify the number of images to produce along the circular path.

                        **Proportion of Circle:**  
                        Sets the proportion of the circle to cover during image generation. 
                        Ranges from 0 to 360 degrees. 
                        Using a high step count with a small number of degrees allows you to explore very subtle image transformations.
                        """)
        with gr.Row():
            with gr.Column():
                prompt_circular = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_images_circular = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Steps around the Circle")

                with gr.Row():
                    degree_circular = gr.Slider(minimum=0, maximum=360, step=1, value=360, label="Proportion of Circle", info="Enter the value in degrees")
                    step_size_circular = gr.Textbox(label="Step Size", value=360/5)

                num_inference_steps_circular = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
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
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        This tab generates noise patterns for two text prompts and then interpolates between them, gradually transforming from the first to the second. 
                        With a large number of perturbation steps the transformation is very gradual and makes a nice animation.
                        
                        **Additional Controls:**  
                        **Number of Interpolation Steps:**  
                        Defines the number of intermediate images to generate between the two prompts.
                        """)
        with gr.Row():
            with gr.Column():
                promptA = gr.Textbox(lines=1, label="First Prompt", value="Self-portrait oil painting, a beautiful man with golden hair, 8k")
                promptB = gr.Textbox(lines=1, label="Second Prompt", value="Self-portrait oil painting, a beautiful woman with golden hair, 8k")
                num_images_interpolate = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Interpolation Steps")
                num_inference_steps_interpolate = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
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
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        This tab explores how perturbations in a local region of the initial latent noise impact the target image. 
                        A small perturbation to the initial latent noise gets carried through the denoising process, demonstrating the global effect it can produce. 
                        
                        **Additional Controls:**  
                        You can adjust the perturbation through the X, Y, height, and width controls. 
                        """)
        with gr.Row():
            with gr.Column():
                prompt_poke = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_poke = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")

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
    
    with gr.Tab("Negative"):
        gr.Markdown("Observe the effect of negative prompts.")
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        Negative prompts steer images away from unwanted features. 
                        For example, “red” as a negative prompt makes the generated image unlikely to have reddish hues. 
                        """)
        with gr.Row():
            with gr.Column():
                prompt_negative = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                neg_prompt = gr.Textbox(lines=1, label="Negative Prompt", value="Yellow")
                num_inference_steps_negative = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")

                with gr.Row():
                    seed_negative = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_negative = gr.Plot(value=generate_seed_vis(14), label="Seed")
                
                generate_images_button_negative = gr.Button("Generate Images")

            with gr.Column():
                images_output_negative = gr.Image(label="Image")
                images_neg_output_negative = gr.Image(label="Image with Negative Prompt")
                zip_output_negative = gr.File(label="Download ZIP")

    seed_negative.change(fn=generate_seed_vis, inputs=[seed_negative], outputs=[seed_vis_negative])
    generate_images_button_negative.click(fn=display_negative_images, inputs=[prompt_negative, seed_negative, num_inference_steps_negative, neg_prompt], outputs=[images_output_negative, images_neg_output_negative, zip_output_negative])

    with gr.Tab("Guidance"):
        gr.Markdown("Observe the effect of different guidance scales.")
        with gr.Accordion("About", open=False):
            gr.Markdown("""
                        Guidance is responsible for making the target image adhere to the prompt. 
                        A higher value enforces this relation, whereas a lower value does not. 
                        For example, a guidance scale of 1 produces a distorted grayscale image, whereas 50 produces a distorted, oversaturated image. 
                        The default value of 8 produces normal-looking images that reasonably adhere to the prompt.
                        """)
        with gr.Row():
            with gr.Column():
                prompt_guidance = gr.Textbox(lines=1, label="Prompt", value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
                num_inference_steps_guidance = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                guidance_scale_values = gr.Textbox(lines=1, value="1, 8, 20, 30", label="Guidance Scale Values")

                with gr.Row():
                    seed_guidance = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_guidance = gr.Plot(value=generate_seed_vis(14), label="Seed")

                generate_images_button_guidance = gr.Button("Generate Images")

            with gr.Column():
                images_output_guidance = gr.Gallery(label="Images", selected_index=0)
                zip_output_guidance = gr.File(label="Download ZIP")

    generate_images_button_guidance.click(fn=display_guidance_images, inputs=[prompt_guidance, seed_guidance, num_inference_steps_guidance, guidance_scale_values], outputs=[images_output_guidance, zip_output_guidance])
    seed_guidance.change(fn=generate_seed_vis, inputs=[seed_guidance], outputs=[seed_vis_guidance])

    with gr.Tab("Inpainting"):
        gr.Markdown("Inpaint the image based on the prompt.")
        with gr.Accordion("About", open=False):
            gr.Markdown("Unlike poke, which globally alters the target image via a perturbation in the initial latent noise, inpainting alters just the region of the perturbation and allows us to specify the change we want to make.")
        with gr.Row():
            with gr.Column():
                uploaded_img_inpaint = gr.Image(source='upload', tool='sketch', type="pil", label="Upload")
                prompt_inpaint = gr.Textbox(lines=1, label="Prompt", value="A apple fruit")
                num_inference_steps_inpaint = gr.Slider(minimum=2, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                
                with gr.Row():
                    seed_inpaint = gr.Slider(minimum=0, maximum=100, step=1, value=14, label="Seed")
                    seed_vis_inpaint= gr.Plot(value=generate_seed_vis(14), label="Seed")

                inpaint_button = gr.Button("Inpaint") 
                
            with gr.Column():
                images_output_inpaint = gr.Image(label="Output")
                zip_output_inpaint = gr.File(label="Download ZIP")

    inpaint_button.click(fn=inpaint, inputs=[uploaded_img_inpaint, num_inference_steps_inpaint, seed_inpaint, prompt_inpaint], outputs=[images_output_inpaint, zip_output_inpaint])
    seed_inpaint.change(fn=generate_seed_vis, inputs=[seed_inpaint], outputs=[seed_vis_inpaint])

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