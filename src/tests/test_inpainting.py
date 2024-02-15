import gradio as gr
from src.util.params import *
from src.pipelines.inpainting import *

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            img = gr.Sketchpad(sources='upload', type="pil", label="Upload", brush=gr.Brush(colors=["#ffff00"]))
            prmpt = gr.Textbox(label="Prompt", value=prompt)
            inf = gr.Number(label="Number of Inference Steps", value=num_inference_steps)
            sd = gr.Number(label="Seed", value=seed)
            btn = gr.Button("Inpaint") 
            
        with gr.Column():
            image_out = gr.Image(label="Output")
            zp = gr.File(label="Download Zip")      

    btn.click(fn=inpaint, inputs=[img, inf, sd, prmpt], outputs=[image_out, zp])

demo.launch()