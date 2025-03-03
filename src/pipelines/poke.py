import os
import gradio as gr
from src.util.base import *
from src.util.params import *
from PIL import Image, ImageDraw
from src.util.session import session_manager


def visualize_poke(
    pokeX, pokeY, pokeHeight, pokeWidth, imageHeight=imageHeight, imageWidth=imageWidth, request: gr.Request = None
):
    if (
        (pokeX - pokeWidth // 2 < 0)
        or (pokeX + pokeWidth // 2 > imageWidth // 8)
        or (pokeY - pokeHeight // 2 < 0)
        or (pokeY + pokeHeight // 2 > imageHeight // 8)
    ):
        gr.Warning("Modification outside image")
    shape = [
        (pokeX * 8 - pokeWidth * 8 // 2, pokeY * 8 - pokeHeight * 8 // 2),
        (pokeX * 8 + pokeWidth * 8 // 2, pokeY * 8 + pokeHeight * 8 // 2),
    ]

    blank = Image.new("RGB", (imageWidth, imageHeight))
    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    original_path = session_dir / "original.png"
    poked_path = session_dir / "poked.png"

    if original_path.exists() and poked_path.exists():
        oImg = Image.open(original_path)
        pImg = Image.open(poked_path)
    else:
        oImg = blank
        pImg = blank

    oRec = ImageDraw.Draw(oImg)
    pRec = ImageDraw.Draw(pImg)

    oRec.rectangle(shape, outline="white")
    pRec.rectangle(shape, outline="white")

    return oImg, pImg


def display_poke_images(
    prompt,
    seed,
    num_inference_steps,
    poke=False,
    pokeX=None,
    pokeY=None,
    pokeHeight=None,
    pokeWidth=None,
    intermediate=False,
    progress=gr.Progress(),
    request: gr.Request = None
):
    text_embeddings = get_text_embeddings(prompt)
    latents, modified_latents = generate_modified_latents(
        poke, seed, pokeX, pokeY, pokeHeight, pokeWidth
    )

    progress(0)
    images = generate_images(
        latents, text_embeddings, num_inference_steps, intermediate=intermediate
    )

    session_dir = session_manager.get_session_path(request.session_hash if request else "default")
    
    if not intermediate:
        images.save(session_dir / "original.png")

    if poke:
        progress(0.5)
        modImages = generate_images(
            modified_latents,
            text_embeddings,
            num_inference_steps,
            intermediate=intermediate,
        )

        if not intermediate:
            modImages.save(session_dir / "poked.png")
    else:
        modImages = None

    return images, modImages


__all__ = ["display_poke_images", "visualize_poke"]
