"""
Parameter configurations for the Diffusion Demo application.

This module defines model paths, default parameters, and initializes the core
components of the Stable Diffusion pipeline. It serves as a central configuration
point for model settings, default prompt values, image dimensions, and safety filters.
"""

import torch
import secrets
from gradio.networking import setup_tunnel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LCMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# Set device based on available hardware
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration for model loading
isLCM = False  # Whether to use Latent Consistency Models scheduler
HF_ACCESS_TOKEN = ""  # Hugging Face access token (empty = use public models)

# Model paths
model_path = "Lykon/dreamshaper-8"  # Main diffusion model
inpaint_model_path = "Lykon/dreamshaper-8-inpainting"  # Model for inpainting

# Default prompts
prompt = "close-up view of a kindly old man wearing a tartan cap, standing under a tree"  # Default generation prompt
promptA = "close-up view of a kindly old man wearing a tartan cap, standing under a tree" # First interpolation prompt
promptB = "close-up view of a kindly old woman wearing a tartan cap, standing under a tree" # Second interpolation prompt

# Content safety settings
negative_prompt = "(nsfw), nude, naked, breasts, underwear, pornography, hentai, sexual, explicit, lewd, obscene, indecent, gore, blood, injury, mutilation, corpse, violence, weapon, gun, knife, killing, death, torture, drugs, narcotics, smoking, alcohol"
bad_concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick', 'vagina', 'naked person (approximation)', 'explicit content', 'uncensored', 'fuck', 'nipples', 'nipples (approximation)', 'naked breasts', 'areola']

# Default generation parameters
num_images = 5  # Number of images to generate in multi-image tabs
degree = 360  # Default degrees for circular interpolation
perturbation_size = 0.1  # Default size of perturbations in latent space
num_inference_steps = 8  # Default number of denoising steps
seed = 69420  # Default random seed for reproducibility

# Image quality and guidance settings
guidance_scale = 8  # Default classifier-free guidance scale
guidance_values = "1, 8, 20"  # Default values for guidance comparison

# Intermediate images and regional modification defaults
intermediate = True  # Whether to save intermediate denoising steps
pokeX, pokeY = 256, 256  # Center point for region modification
pokeHeight, pokeWidth = 128, 128  # Size of the region modification
imageHeight, imageWidth = 512, 512  # Default output image dimensions

# Initialize model components
# Load the CLIP tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(
    torch_device
)

# Initialize the appropriate scheduler based on configuration
if isLCM:
    scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler")
else:
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

# Load the U-Net model for noise prediction
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(
    torch_device
)

# Load the Variational Autoencoder for image encoding/decoding
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

# Load safety components
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", use_safetensors=True)
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the full Stable Diffusion pipeline
pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    unet=unet,
    scheduler=scheduler,
    vae=vae,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,    
).to(torch_device)

# dash_tunnel = setup_tunnel("0.0.0.0", 8000, secrets.token_urlsafe(32), None)

# Session cleanup settings
cleanup_interval = 30  # Run user session directory cleanup every 30 seconds
cleanup_threshold = 30  # Delete user session directory if not accessed for 30 seconds

# Export all public variables from this module
__all__ = [
    "prompt",
    "negative_prompt",
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
    "guidance_values",
    "HF_ACCESS_TOKEN",
    "model_path",
    "inpaint_model_path",
    # "dash_tunnel",
    "pipe",
    "cleanup_interval",
    "cleanup_threshold",
    "safety_checker",
    "feature_extractor",
    "bad_concepts"
]
