import os, time
import gradio as gr
from src.util import *
from src.pipelines import *
from threading import Thread
from serve import run_flask_server
from src.util.session import session_manager

with gr.Blocks(css="#step_size_circular {background-color: #666666} #step_size_circular textarea {background-color: #666666}") as demo:
    gr.Markdown("## Stable Diffusion Demo")
    session_hash_state = gr.State("")

    with gr.Tab("Latent Space"):

        with gr.TabItem("Beginner"):
            gr.Markdown("Generate images from text.")

            with gr.Row():
                with gr.Column():
                    prompt_beginner = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )

                    with gr.Row():
                        seed_beginner = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_beginner = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_beginner = gr.Button("Generate Image")

                with gr.Column():
                    images_output_beginner = gr.Image(label="Image")

        @generate_images_button_beginner.click(
            inputs=[prompt_beginner, seed_beginner],
            outputs=[images_output_beginner],
        )
        def generate_images_wrapper(
            prompt, seed, progress=gr.Progress(), request: gr.Request = None
        ):
            images, _ = display_poke_images(
                prompt, seed, num_inference_steps=8, poke=False, intermediate=False, request=request
            )
            return images

        seed_beginner.change(
            fn=generate_seed_vis, inputs=[seed_beginner], outputs=[seed_vis_beginner]
        )

        with gr.TabItem("Denoising"):
            gr.Markdown("Observe the intermediate images during denoising.")
            gr.HTML(read_html("DiffusionDemo/html/denoising.html"))

            with gr.Row():
                with gr.Column():
                    prompt_denoise = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_inference_steps_denoise = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps",
                    )

                    with gr.Row():
                        seed_denoise = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_denoise = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_denoise = gr.Button("Generate Images")

                with gr.Column():
                    images_output_denoise = gr.Gallery(label="Images", selected_index=0)
                    gif_denoise = gr.Image(label="GIF")
                    zip_output_denoise = gr.File(label="Download ZIP")

        @generate_images_button_denoise.click(
            inputs=[prompt_denoise, seed_denoise, num_inference_steps_denoise],
            outputs=[images_output_denoise, gif_denoise, zip_output_denoise],
        )
        def generate_images_wrapper(
            prompt, seed, num_inference_steps, progress=gr.Progress(), request: gr.Request = None
        ):
            images, _ = display_poke_images(
                prompt, seed, num_inference_steps, poke=False, intermediate=True, request=request
            )
            fname = "denoising"
            tab_config = {
                "Tab": "Denoising",
                "Prompt": prompt,
                "Number of Inference Steps": num_inference_steps,
                "Seed": seed,
            }
            zip_path = export_as_zip(images, fname, tab_config, request=request)
            progress(1, desc="Exporting as gif")
            gif_path = export_as_gif(images, filename="denoising.gif", request=request)
            return images, gif_path, zip_path

        seed_denoise.change(
            fn=generate_seed_vis, inputs=[seed_denoise], outputs=[seed_vis_denoise]
        )

        with gr.TabItem("Seeds"):
            gr.Markdown(
                "Understand how different starting points in latent space can lead to different images."
            )
            gr.HTML(read_html("DiffusionDemo/html/seeds.html"))

            with gr.Row():
                with gr.Column():
                    prompt_seed = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_images_seed = gr.Slider(
                        minimum=1, maximum=100, step=1, value=5, label="Number of Seeds"
                    )
                    num_inference_steps_seed = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )
                    generate_images_button_seed = gr.Button("Generate Images")

                with gr.Column():
                    images_output_seed = gr.Gallery(label="Images", selected_index=0)
                    zip_output_seed = gr.File(label="Download ZIP")

        generate_images_button_seed.click(
            fn=display_seed_images,
            inputs=[prompt_seed, num_inference_steps_seed, num_images_seed],
            outputs=[images_output_seed, zip_output_seed],
        )

        with gr.TabItem("Perturbations"):
            gr.Markdown("Explore different perturbations from a point in latent space.")
            gr.HTML(read_html("DiffusionDemo/html/perturbations.html"))

            with gr.Row():
                with gr.Column():
                    prompt_perturb = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_images_perturb = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Perturbations",
                    )
                    perturbation_size_perturb = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.1,
                        label="Perturbation Size",
                    )
                    num_inference_steps_perturb = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_perturb = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_perturb = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_perturb = gr.Button("Generate Images")

                with gr.Column():
                    images_output_perturb = gr.Gallery(label="Image", selected_index=0)
                    zip_output_perturb = gr.File(label="Download ZIP")

        generate_images_button_perturb.click(
            fn=display_perturb_images,
            inputs=[
                prompt_perturb,
                seed_perturb,
                num_inference_steps_perturb,
                num_images_perturb,
                perturbation_size_perturb,
            ],
            outputs=[images_output_perturb, zip_output_perturb],
        )
        seed_perturb.change(
            fn=generate_seed_vis, inputs=[seed_perturb], outputs=[seed_vis_perturb]
        )

        with gr.TabItem("Circular"):
            gr.Markdown(
                "Generate a circular path in latent space and observe how the images vary along the path."
            )
            gr.HTML(read_html("DiffusionDemo/html/circular.html"))

            with gr.Row():
                with gr.Column():
                    prompt_circular = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_images_circular = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Steps around the Circle",
                    )

                    with gr.Row():
                        start_degree_circular = gr.Slider(
                            minimum=0,
                            maximum=360,
                            step=1,
                            value=0,
                            label="Start Angle",
                            info="Enter the value in degrees",
                        )
                        end_degree_circular = gr.Slider(
                            minimum=0,
                            maximum=360,
                            step=1,
                            value=360,
                            label="End Angle",
                            info="Enter the value in degrees",
                        )
                        step_size_circular = gr.Textbox(
                            label="Step Size", value=360 / 5,
                            elem_id="step_size_circular"
                        )

                    num_inference_steps_circular = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_circular = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_circular = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_circular = gr.Button("Generate Images")

                with gr.Column():
                    images_output_circular = gr.Gallery(label="Image", selected_index=0)
                    gif_circular = gr.Image(label="GIF")
                    zip_output_circular = gr.File(label="Download ZIP")

        num_images_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        start_degree_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        end_degree_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        generate_images_button_circular.click(
            fn=display_circular_images,
            inputs=[
                prompt_circular,
                seed_circular,
                num_inference_steps_circular,
                num_images_circular,
                start_degree_circular,
                end_degree_circular,
            ],
            outputs=[images_output_circular, gif_circular, zip_output_circular],
        )
        seed_circular.change(
            fn=generate_seed_vis, inputs=[seed_circular], outputs=[seed_vis_circular]
        )

        with gr.TabItem("Poke"):
            gr.Markdown("Perturb a region in the image and observe the effect.")
            gr.HTML(read_html("DiffusionDemo/html/poke.html"))

            with gr.Row():
                with gr.Column():
                    prompt_poke = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_inference_steps_poke = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_poke = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_poke = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    pokeX = gr.Slider(
                        label="pokeX",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=32,
                        info="X coordinate of poke center",
                    )
                    pokeY = gr.Slider(
                        label="pokeY",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=32,
                        info="Y coordinate of poke center",
                    )
                    pokeHeight = gr.Slider(
                        label="pokeHeight",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=8,
                        info="Height of the poke",
                    )
                    pokeWidth = gr.Slider(
                        label="pokeWidth",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=8,
                        info="Width of the poke",
                    )

                    generate_images_button_poke = gr.Button("Generate Images")

                with gr.Column():
                    original_images_output_poke = gr.Image(
                        value=visualize_poke(32, 32, 8, 8)[0], label="Original Image"
                    )
                    poked_images_output_poke = gr.Image(
                        value=visualize_poke(32, 32, 8, 8)[1], label="Poked Image"
                    )
                    zip_output_poke = gr.File(label="Download ZIP")

        pokeX.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth],
            outputs=[original_images_output_poke, poked_images_output_poke],
        )
        pokeY.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth],
            outputs=[original_images_output_poke, poked_images_output_poke],
        )
        pokeHeight.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth],
            outputs=[original_images_output_poke, poked_images_output_poke],
        )
        pokeWidth.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth],
            outputs=[original_images_output_poke, poked_images_output_poke],
        )
        seed_poke.change(
            fn=generate_seed_vis, inputs=[seed_poke], outputs=[seed_vis_poke]
        )

        @generate_images_button_poke.click(
            inputs=[
                prompt_poke,
                seed_poke,
                num_inference_steps_poke,
                pokeX,
                pokeY,
                pokeHeight,
                pokeWidth,
            ],
            outputs=[
                original_images_output_poke,
                poked_images_output_poke,
                zip_output_poke,
            ],
        )
        def generate_images_wrapper(
            prompt,
            seed,
            num_inference_steps,
            pokeX=pokeX,
            pokeY=pokeY,
            pokeHeight=pokeHeight,
            pokeWidth=pokeWidth,
            request: gr.Request = None
        ):
            _, _ = display_poke_images(
                prompt,
                seed,
                num_inference_steps,
                poke=True,
                pokeX=pokeX,
                pokeY=pokeY,
                pokeHeight=pokeHeight,
                pokeWidth=pokeWidth,
                intermediate=False,
                request=request
            )
            images, modImages = visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, request=request)
            fname = "poke"
            tab_config = {
                "Tab": "Poke",
                "Prompt": prompt,
                "Number of Inference Steps per Image": num_inference_steps,
                "Seed": seed,
                "PokeX": pokeX,
                "PokeY": pokeY,
                "PokeHeight": pokeHeight,
                "PokeWidth": pokeWidth,
            }
            imgs_list = []
            imgs_list.append((images, "Original Image"))
            imgs_list.append((modImages, "Poked Image"))
            
            zip_path = export_as_zip(imgs_list, fname, tab_config, request=request)
            return images, modImages, zip_path

        with gr.TabItem("Guidance"):
            gr.Markdown("Observe the effect of different guidance scales.")
            gr.HTML(read_html("DiffusionDemo/html/guidance.html"))

            with gr.Row():
                with gr.Column():
                    prompt_guidance = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    num_inference_steps_guidance = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )
                    guidance_scale_values = gr.Textbox(
                        lines=1, value="1, 8, 20, 30", label="Guidance Scale Values"
                    )

                    with gr.Row():
                        seed_guidance = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_guidance = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_guidance = gr.Button("Generate Images")

                with gr.Column():
                    images_output_guidance = gr.Gallery(
                        label="Images", selected_index=0
                    )
                    zip_output_guidance = gr.File(label="Download ZIP")

        generate_images_button_guidance.click(
            fn=display_guidance_images,
            inputs=[
                prompt_guidance,
                seed_guidance,
                num_inference_steps_guidance,
                guidance_scale_values,
            ],
            outputs=[images_output_guidance, zip_output_guidance],
        )
        seed_guidance.change(
            fn=generate_seed_vis, inputs=[seed_guidance], outputs=[seed_vis_guidance]
        )

        with gr.TabItem("Inpainting"):
            gr.Markdown("Inpaint the image based on the prompt.")
            gr.HTML(read_html("DiffusionDemo/html/inpainting.html"))

            with gr.Row():
                with gr.Column():
                    uploaded_img_inpaint = gr.Sketchpad(
                        sources="upload", brush=gr.Brush(colors=["#ffff00"]), type="pil", label="Upload", height=imageHeight
                    )
                    prompt_inpaint = gr.Textbox(
                        lines=1, label="Prompt", value="sunglasses"
                    )
                    num_inference_steps_inpaint = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_inpaint = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_inpaint = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    inpaint_button = gr.Button("Inpaint")

                with gr.Column():
                    images_output_inpaint = gr.Image(label="Output")
                    zip_output_inpaint = gr.File(label="Download ZIP")

        inpaint_button.click(
            fn=inpaint,
            inputs=[
                uploaded_img_inpaint,
                num_inference_steps_inpaint,
                seed_inpaint,
                prompt_inpaint,
            ],
            outputs=[images_output_inpaint, zip_output_inpaint],
        )
        seed_inpaint.change(
            fn=generate_seed_vis, inputs=[seed_inpaint], outputs=[seed_vis_inpaint]
        )

    with gr.Tab("CLIP Space"):
        with gr.TabItem("Embeddings"):
            with gr.Row():
                output = gr.HTML(value="Loading...", elem_id="embedding-html")

            with gr.Row():
                word2add_rem = gr.Textbox(lines=1, label="Add/Remove word")
                word2change = gr.Textbox(lines=1, label="Change image for word")
                clear_words_button = gr.Button(value="Clear words")

            with gr.Row():
                with gr.Column(scale=1):
                    word_input = gr.Textbox(
                        label="Visualize embedding for word", lines=1
                    )

                with gr.Column(scale=1):
                    embedding_visualization = gr.Image(
                        type="pil", interactive=False, height="6vw"
                    )

            with gr.Row():
                gallery = gr.Gallery(
                    label="Images of words",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )

            with gr.Accordion("Custom Semantic Dimensions", open=False):
                with gr.Row():
                    axis_name_1 = gr.Textbox(label="Axis name", value="gender")
                    which_axis_1 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_1"],
                        label="Axis direction",
                    )
                    from_words_1 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="prince husband father son uncle",
                    )
                    to_words_1 = gr.Textbox(
                        lines=1,
                        label="Negative",
                        value="princess wife mother daughter aunt",
                    )
                    submit_1 = gr.Button("Submit")

                with gr.Row():
                    axis_name_2 = gr.Textbox(label="Axis name", value="age")
                    which_axis_2 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_2"],
                        label="Axis direction",
                    )
                    from_words_2 = gr.Textbox(
                        lines=1, label="Positive", value="man woman king queen father"
                    )
                    to_words_2 = gr.Textbox(
                        lines=1, label="Negative", value="boy girl prince princess son"
                    )
                    submit_2 = gr.Button("Submit")

                with gr.Row():
                    axis_name_3 = gr.Textbox(label="Axis name", value="residual")
                    which_axis_3 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_3"],
                        label="Axis direction",
                    )
                    from_words_3 = gr.Textbox(lines=1, label="Positive")
                    to_words_3 = gr.Textbox(lines=1, label="Negative")
                    submit_3 = gr.Button("Submit")

                with gr.Row():
                    axis_name_4 = gr.Textbox(label="Axis name", value="number")
                    which_axis_4 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_4"],
                        label="Axis direction",
                    )
                    from_words_4 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="boys girls cats puppies computers",
                    )
                    to_words_4 = gr.Textbox(
                        lines=1, label="Negative", value="boy girl cat puppy computer"
                    )
                    submit_4 = gr.Button("Submit")

                with gr.Row():
                    axis_name_5 = gr.Textbox(label="Axis name", value="royalty")
                    which_axis_5 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_5"],
                        label="Axis direction",
                    )
                    from_words_5 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="king queen prince princess duchess",
                    )
                    to_words_5 = gr.Textbox(
                        lines=1, label="Negative", value="man woman boy girl woman"
                    )
                    submit_5 = gr.Button("Submit")

                with gr.Row():
                    axis_name_6 = gr.Textbox(label="Axis name")
                    which_axis_6 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_6"],
                        label="Axis direction",
                    )
                    from_words_6 = gr.Textbox(lines=1, label="Positive")
                    to_words_6 = gr.Textbox(lines=1, label="Negative")
                    submit_6 = gr.Button("Submit")


        def load_user_html(request: gr.Request):
            flask_url, session_hash, is_new = init_user_session(request)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            if is_new:
                gr.Info("New session initialized.")

            gallery_images = load_user_gallery(session_hash)

            return html_content, session_hash, gallery_images

        demo.load(load_user_html, None, [output, session_hash_state, gallery])

        @word2add_rem.submit(
            inputs=[word2add_rem, session_hash_state],
            outputs=[output, word2add_rem, gallery],
        )
        def add_rem_word_handler(words, session_hash):
            flask_url = add_rem_word_user(words, session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return html_content, "", gallery_images

        @word2change.submit(
            inputs=[word2change, session_hash_state], outputs=[output, word2change, gallery]
        )
        def change_word_handler(word, session_hash):
            flask_url = change_word_user(word, session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return html_content, "", gallery_images

        @clear_words_button.click(inputs=[session_hash_state], outputs=[output, gallery])
        def clear_words_handler(session_hash):
            clear_url = clear_words_user(session_hash)
            html_content = f"""<iframe id="html-frame" src="{clear_url}" style="width:100%; height:700px;"></iframe>"""
            gallery_images = load_user_gallery(session_hash)
            return html_content, gallery_images

        @submit_1.click(
            inputs=[
                axis_name_1,
                which_axis_1,
                from_words_1,
                to_words_1,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_1"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_2.click(
            inputs=[
                axis_name_2,
                which_axis_2,
                from_words_2,
                to_words_2,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_3,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_2"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_3.click(
            inputs=[
                axis_name_3,
                which_axis_3,
                from_words_3,
                to_words_3,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_3"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_4.click(
            inputs=[
                axis_name_4,
                which_axis_4,
                from_words_4,
                to_words_4,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_4"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_5.click(
            inputs=[
                axis_name_5,
                which_axis_5,
                from_words_5,
                to_words_5,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_5"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_6.click(
            inputs=[
                axis_name_6,
                which_axis_6,
                from_words_6,
                to_words_6,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_5,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_6"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
            )

        @word_input.submit(
            inputs=[word_input, session_hash_state],
            outputs=[embedding_visualization, word_input, gallery],
        )
        def handle_word_visualization(word, session_hash):
            if not word.strip():
                return None, "", load_user_gallery(session_hash)

            emb_viz, generated_img, label = generate_word_embedding_visualization(
                word, session_hash
            )

            if "not in examples" in label:
                gr.Warning(
                    f"'{word}' not in examples. Please add it first using the Add/Remove word field."
                )
                return None, "", load_user_gallery(session_hash)

            return emb_viz, "", load_user_gallery(session_hash)

        with gr.TabItem("Interpolate"):
            gr.Markdown(
                "Interpolate between the first and the second prompt, and observe how the output changes."
            )
            gr.HTML(read_html("DiffusionDemo/html/interpolate.html"))

            with gr.Row():
                with gr.Column():
                    promptA = gr.Textbox(
                        lines=1,
                        label="First Prompt",
                        value="Self-portrait oil painting, a beautiful man with golden hair, 8k",
                    )
                    promptB = gr.Textbox(
                        lines=1,
                        label="Second Prompt",
                        value="Self-portrait oil painting, a beautiful woman with golden hair, 8k",
                    )
                    num_images_interpolate = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Interpolation Steps",
                    )
                    num_inference_steps_interpolate = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_interpolate = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_interpolate = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_interpolate = gr.Button("Generate Images")

                with gr.Column():
                    images_output_interpolate = gr.Gallery(
                        label="Interpolated Images", selected_index=0
                    )
                    gif_interpolate = gr.Image(label="GIF")
                    zip_output_interpolate = gr.File(label="Download ZIP")

        generate_images_button_interpolate.click(
            fn=display_interpolate_images,
            inputs=[
                seed_interpolate,
                promptA,
                promptB,
                num_inference_steps_interpolate,
                num_images_interpolate,
            ],
            outputs=[
                images_output_interpolate,
                gif_interpolate,
                zip_output_interpolate,
            ],
        )
        seed_interpolate.change(
            fn=generate_seed_vis,
            inputs=[seed_interpolate],
            outputs=[seed_vis_interpolate],
        )

        with gr.TabItem("Negative"):
            gr.Markdown("Observe the effect of negative prompts.")
            gr.HTML(read_html("DiffusionDemo/html/negative.html"))

            with gr.Row():
                with gr.Column():
                    prompt_negative = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    )
                    neg_prompt = gr.Textbox(
                        lines=1, label="Negative Prompt", value="Yellow"
                    )
                    num_inference_steps_negative = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    with gr.Row():
                        seed_negative = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_negative = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    generate_images_button_negative = gr.Button("Generate Images")

                with gr.Column():
                    images_output_negative = gr.Image(
                        label="Image without Negative Prompt"
                    )
                    images_neg_output_negative = gr.Image(
                        label="Image with Negative Prompt"
                    )
                    zip_output_negative = gr.File(label="Download ZIP")

        seed_negative.change(
            fn=generate_seed_vis, inputs=[seed_negative], outputs=[seed_vis_negative]
        )
        generate_images_button_negative.click(
            fn=display_negative_images,
            inputs=[
                prompt_negative,
                seed_negative,
                num_inference_steps_negative,
                neg_prompt,
            ],
            outputs=[
                images_output_negative,
                images_neg_output_negative,
                zip_output_negative,
            ],
        )

    with gr.Tab("Credits"):
        gr.Markdown("""
                    Author: Adithya Kameswara Rao, Carnegie Mellon University.

                    Advisor: David S. Touretzky, Carnegie Mellon University.

                    This work was funded by a grant from NEOM Company, and by National Science Foundation award IIS-2112633.
                    """)
    
def run_gradio():
    demo.queue(
        default_concurrency_limit=2,  
        max_size=4,  
        api_open=False 
    )
    os.makedirs("outputs", exist_ok=True)
    _, _, public_url = demo.launch(
        share=True,
        max_threads=8  
    )
    return public_url


if __name__ == "__main__":
    session_manager.start_cleanup_thread()

    flask_thread = Thread(target=run_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    
    time.sleep(2)
    
    try:
        run_gradio()
    except KeyboardInterrupt:
        print("Server closed")
        session_manager.stop_cleanup_thread()
