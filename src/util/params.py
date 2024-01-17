import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler, LMSDiscreteScheduler
import tntn

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

HF_ACCESS_TOKEN = ""                                                           # Add your HuggingFace access token

model_path = "segmind/tiny-sd"                                                 # Huggingface model path
imageHeight, imageWidth = 512, 512                                             # Image size
guidance_scale = 8                                                             # Guidance scale

prompt = "red balloon in the sky"
num_images = 5                                                                  # Number of images to generate
differentiation = 0.1                                                           # Differentiation factor
num_inference_steps = 8                                                        # 20+ for good results
seed = 69420                                                                    # Any integer
intermediate = False                           

pokeX, pokeY = 256, 256                                                         # Poke coordinates
pokeHeight, pokeWidth = 128, 128                                                # Poke size

promptA = "a car driving on a highway in a city"
promptB = "a truck driving on a highway in a city"

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)
scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

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