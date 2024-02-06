import torch
import secrets
from gradio.networking import setup_tunnel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

isLCM = False
HF_ACCESS_TOKEN = ""        

model_path = "segmind/small-sd"    
prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
promptA = "Self-portrait oil painting, a beautiful man with golden hair, 8k"
promptB = "Self-portrait oil painting, a beautiful woman with golden hair, 8k"

num_images = 5                                                                  
degree = 360  
perturbation_size = 0.1                                                            
num_inference_steps = 8                                                        
seed = 69420                                                              

guidance_scale = 8                                                                 
intermediate = True                           
pokeX, pokeY = 256, 256                                                         
pokeHeight, pokeWidth = 128, 128                                                
imageHeight, imageWidth = 512, 512  

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)

if isLCM:
    scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler")
else:
    scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

dash_tunnel = setup_tunnel('0.0.0.0', 8000, secrets.token_urlsafe(32), None)

__all__ = [
    "prompt", 
    "num_images", 
    "degree",
    "perturbation_size", 
    "num_inference_steps", 
    "seed", 
    "intermediate", 
    "pokeX", 
    "pokeY", 
    "pokeHeight", 
    "pokeWidth", 
    "promptA", 
    "promptB", 
    "tokenizer", 
    "text_encoder", 
    "scheduler", 
    "unet", 
    "vae", 
    "torch_device", 
    "imageHeight", 
    "imageWidth", 
    "guidance_scale", 
    "HF_ACCESS_TOKEN",
    "model_path",
    "dash_tunnel"
]