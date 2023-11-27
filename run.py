import gradio as gr
from src.main.poke import visualize_poke, display_images
from src.main.prompt_interpolation import display_prompt_images
from src.main.similar_image_generation import display_similar_images
from src.main.dataset_peek import retrieve_images

import os
import torch
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

HF_ACCESS_TOKEN = ""                                                           # Add your HuggingFace access token

model_path = "/home/akameswa/research/models/tiny-sd"                          # Huggingface model path
imageHeight, imageWidth = 512, 512                                             # Image size
guidance_scale = 8                                                             # Guidance scale

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)
scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion Demo")
    with gr.Tab("PokE"):
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
                prompt_0 = gr.Textbox(lines=1, label="Prompt", value="dog in car")
                num_inference_steps_0 = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Inference Steps per Image")
                seed_0 = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_0 = gr.Button("Generate Images")
            
            with gr.Column():
                original_images_output_0 = gr.Gallery(label="Original Images", selected_index=0)
                poked_images_output_0 = gr.Gallery(label="Poked Images", selected_index=0)

    with gr.Tab("Prompt Interpolation"):
        with gr.Row():
            with gr.Column():
                promptA = gr.Textbox(lines=1, label="Prompt from", value="car driving on a highway in a city")
                num_images_1 = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                num_inference_steps_1 = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                seed_1 = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                
            with gr.Column():
                promptB = gr.Textbox(lines=1, label="Prompt to", value="truck driving on a highway in a city")                
        
        with gr.Row():
            generate_images_button_1 = gr.Button("Generate Images")

        with gr.Row():
            images_output_1 = gr.Gallery(label="Interpolated Images", selected_index=0)

    with gr.Tab("Similar Image Generation"):
        with gr.Row():
            with gr.Column():
                prompt_2 = gr.Textbox(lines=1, label="Prompt", value="dog in car")
                num_images_2 = gr.Slider(minimum=0, maximum=100, step=1, value=5, label="Number of Images")
                differentiation = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Differentiation", info="The higher the differentiation, the more different the images will be")
                num_inference_steps_2 = gr.Slider(minimum=0, maximum=100, step=1, value=8, label="Number of Inference Steps per Image")
                seed_2 = gr.Slider(minimum=0, maximum=100, step=1, value=69, label="Seed")
                generate_images_button_2 = gr.Button("Generate Images")

            with gr.Column():
                images_output_2 = gr.Gallery(label="Image", selected_index=0)

    with gr.Tab("Dataset Peek"):
        with gr.Row():
            with gr.Column():
                prompt_3 = gr.Textbox(lines=1, label="Prompt", value="dog")
                retrieve_images_button = gr.Button("Retrieve Images")
                retrieved_images_output = gr.Gallery(label="Image", selected_index=0)
    
    @visualize_poke_button.click(inputs=[pokeX, pokeY, pokeHeight, pokeWidth], outputs=visualize_poke_output)
    def visualize_poke_wrapper(pokeX, pokeY, pokeHeight, pokeWidth):
        return visualize_poke(pokeX, pokeY, pokeHeight, pokeWidth, imageHeight, imageWidth)

    @generate_images_button_0.click(inputs=[prompt_0, pokeX, pokeY, pokeHeight, pokeWidth, seed_0, num_inference_steps_0], outputs=[original_images_output_0, poked_images_output_0])
    def generate_images_wrapper(prompt, pokeX, pokeY, pokeHeight, pokeWidth, seed, num_inference_steps):
        images, modImages = display_images(prompt, pokeX, pokeY, pokeHeight, pokeWidth, seed, num_inference_steps, tokenizer, text_encoder, unet, scheduler, vae, guidance_scale, torch_device, imageHeight, imageWidth, intermediate=True)
        return images, modImages
    
    @generate_images_button_1.click(inputs=[seed_1, promptA, promptB, num_inference_steps_1, num_images_1], outputs=images_output_1)
    def display_prompt_images_wrapper(seed, promptA, promptB, num_inference_steps, num_images):
        return display_prompt_images(seed, promptA, promptB, num_inference_steps, num_images, imageHeight, imageWidth, guidance_scale, tokenizer, text_encoder, scheduler, unet, vae, torch_device, intermediate=False)
    
    @generate_images_button_2.click(inputs=[prompt_2, seed_2, num_inference_steps_2, num_images_2, differentiation], outputs=images_output_2)
    def display_similar_images_wrapper(prompt, seed, num_inference_steps, num_images, differentiation):
        return display_similar_images(prompt, seed, num_inference_steps, num_images, imageHeight, imageWidth, differentiation, guidance_scale, tokenizer, text_encoder, unet, scheduler, vae, torch_device)
    
    retrieve_images_button.click(fn=retrieve_images, inputs=[prompt_3], outputs=retrieved_images_output)

demo.queue()
demo.launch(share=True)