import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler
import tntn

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

isLCM = False
isLocal = False
HF_ACCESS_TOKEN = ""        

model_path = "segmind/small-sd"    
prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
promptA = "Self-portrait oil painting, a beautiful man with golden hair, 8k"
promptB = "Self-portrait oil painting, a beautiful woman with golden hair, 8k"

num_images = 5                                                                  
differentiation = 360                                                           
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

if isLocal: 
    dash_tunnel = "http://127.0.0.1:8000/"
else:
    dash_tunnel = tntn.bore(8000).tunnel

__all__ = [
    "prompt", 
    "num_images", 
    "differentiation", 
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