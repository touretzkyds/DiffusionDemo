import random
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

import base64
from io import BytesIO
import plotly.express as px
import os

from src.util.base import (
    get_word_embeddings,
    get_concat_embeddings,
    calculate_residual,
    get_axis_embeddings,
    get_user_dir,
    get_user_examples_dir,
    get_user_viz_dir,
)
from src.util.params import num_inference_steps, guidance_scale, tokenizer, pipe
from src.util.clip_config import (
    masculine,
    feminine,
    young,
    old,
    common,
    elite,
    examples,
    axis_names,
    axisMap,
)
from PIL import Image
import time

from serve import get_flask_tunnel_url

age = get_axis_embeddings(young, old)
gender = get_axis_embeddings(masculine, feminine)
royalty = get_axis_embeddings(common, elite)


def generate_examples(
    examples=examples,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
):
    images = []
    for example in examples:
        image = pipe(
            prompt=example,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append("data:image/jpeg;base64, " + encoded_image)
    return images


images = generate_examples()

axis = np.vstack([gender, royalty, age])
axis[1] = calculate_residual(axis, axis_names)

coords = get_concat_embeddings(examples) @ axis.T
coords[:, 1] = 5 * (1.0 - coords[:, 1])

default_examples = examples.copy()
default_images = images.copy()
default_coords = coords.copy()
user_data = {}


def get_safe_filename(word):
    """Convert a word to a safe filename"""
    return "".join([c if c.isalnum() else "_" for c in word])


def generate_user_html(session_hash):
    """Generate the HTML file for a user's session"""
    if not session_hash:
        return None

    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return None

    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)

    user_fig = user_data[session_hash]["fig"]

    user_fig.write_html(
        str_html_path,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )

    print(f"Generated HTML at: {str_html_path}")
    print(f"File exists after generation: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")

    try:
        os.chmod(str_html_path, 0o644)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")

    return html_path


def is_new_session(session_hash):
    """Check if this is a new session"""
    return session_hash not in user_data


def init_user_session(request: gr.Request):
    session_hash = request.session_hash
    if not session_hash:
        session_hash = str(random.randint(10000, 99999))

    print(f"Initializing session for: {session_hash}")

    is_new = is_new_session(session_hash)

    if is_new:
        user_data[session_hash] = {
            "examples": default_examples.copy(),
            "images": default_images.copy(),
            "coords": default_coords.copy(),
            "axis": axis.copy(),
            "axis_names": axis_names.copy(),
        }

        user_fig = px.scatter_3d(
            x=user_data[session_hash]["coords"][:, 0],
            y=user_data[session_hash]["coords"][:, 1],
            z=user_data[session_hash]["coords"][:, 2],
            labels={
                "x": user_data[session_hash]["axis_names"][0],
                "y": user_data[session_hash]["axis_names"][1],
                "z": user_data[session_hash]["axis_names"][2],
            },
            text=user_data[session_hash]["examples"],
            height=750,
        )

        user_fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene_camera=dict(eye=dict(x=2, y=2, z=0.1)),
        )

        user_fig.update_traces(
            hoverinfo="text+x+y+z",
            hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}"
        )
        
        user_data[session_hash]["fig"] = user_fig

        examples_dir = get_user_examples_dir(session_hash)

        if examples_dir:
            base_examples_dir = "DiffusionDemo/images/examples"
            
            for example in user_data[session_hash]["examples"]:
                safe_filename = get_safe_filename(example)
                dest_path = os.path.join(examples_dir, f"{safe_filename}.jpg")
                
                if os.path.exists(dest_path):
                    continue
                
                source_path = os.path.join(base_examples_dir, f"{safe_filename}.jpg")
                
                if os.path.exists(source_path):
                    try:
                        import shutil
                        shutil.copy2(source_path, dest_path)
                        print(f"Copied image for '{example}' from pre-generated source")
                    except Exception as e:
                        print(f"Error copying image for '{example}': {e}")
                else:
                    try:
                        image = pipe(
                            prompt=example,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                        ).images[0]
                        image.save(dest_path, format="JPEG")
                        print(f"Generated new image for '{example}' as pre-generated source was not found")
                    except Exception as e:
                        print(f"Error generating initial image for '{example}': {e}")

    html_path = generate_user_html(session_hash)

    timestamp = int(time.time())
    
    flask_tunnel = get_flask_tunnel_url()
    flask_url = f"{flask_tunnel}/plot/{session_hash}?t={timestamp}"

    return flask_url, session_hash, is_new


def update_user_fig(session_hash):
    user_data[session_hash]["fig"].data[0].x = user_data[session_hash]["coords"][:, 0]
    user_data[session_hash]["fig"].data[0].y = user_data[session_hash]["coords"][:, 1]
    user_data[session_hash]["fig"].data[0].z = user_data[session_hash]["coords"][:, 2]
    user_data[session_hash]["fig"].data[0].text = user_data[session_hash]["examples"]

    user_data[session_hash]["fig"].update_layout(
        scene=dict(
            xaxis_title=user_data[session_hash]["axis_names"][0],
            yaxis_title=user_data[session_hash]["axis_names"][1],
            zaxis_title=user_data[session_hash]["axis_names"][2],
        )
    )
    
    user_data[session_hash]["fig"].update_traces(
        hoverinfo="text+x+y+z",
        hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}"
    )

    html_path = generate_user_html(session_hash)

    timestamp = int(time.time())
    flask_tunnel = get_flask_tunnel_url()
    return f"{flask_tunnel}/plot/{session_hash}?t={timestamp}"

def add_word_user(new_example, session_hash):
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]
    user_images = user_data[session_hash]["images"]
    user_axis = user_data[session_hash]["axis"]

    new_coord = get_concat_embeddings([new_example]) @ user_axis.T
    new_coord[:, 1] = 5 * (1.0 - new_coord[:, 1])
    user_data[session_hash]["coords"] = np.vstack([user_coords, new_coord])

    image = pipe(
        prompt=new_example,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    examples_dir = get_user_examples_dir(session_hash)
    safe_filename = get_safe_filename(new_example)
    image_path = examples_dir / f"{safe_filename}.jpg"
    image.save(str(image_path), format="JPEG")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    user_data[session_hash]["images"].append("data:image/jpeg;base64, " + encoded_image)
    user_data[session_hash]["examples"].append(new_example)

    return update_user_fig(session_hash)


def remove_word_user(word_to_remove, session_hash):
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]
    user_images = user_data[session_hash]["images"]

    examplesMap = {example: index for index, example in enumerate(user_examples)}
    if word_to_remove not in examplesMap:
        return update_user_fig(session_hash)

    index = examplesMap[word_to_remove]

    examples_dir = get_user_examples_dir(session_hash)
    safe_filename = get_safe_filename(word_to_remove)
    image_path = examples_dir / f"{safe_filename}.jpg"
    if image_path.exists():
        try:
            image_path.unlink()
        except Exception as e:
            print(f"Warning: Could not remove image file: {e}")

    viz_dir = get_user_viz_dir(session_hash)
    viz_path = viz_dir / f"{safe_filename}_emb.png"
    if viz_path.exists():
        try:
            viz_path.unlink()
        except Exception as e:
            print(f"Warning: Could not remove visualization file: {e}")

    user_data[session_hash]["coords"] = np.delete(user_coords, index, 0)
    user_data[session_hash]["images"].pop(index)
    user_data[session_hash]["examples"].pop(index)

    return update_user_fig(session_hash)


def add_rem_word_user(new_examples, session_hash):
    new_examples = new_examples.replace(",", " ").split()

    for new_example in new_examples:
        if new_example in user_data[session_hash]["examples"]:
            remove_word_user(new_example, session_hash)
            gr.Info(f"Removed {new_example}")
        else:
            tokens = tokenizer.encode(new_example)
            if len(tokens) != 3:
                gr.Warning(f"{new_example} not found in embeddings")
            else:
                add_word_user(new_example, session_hash)
                gr.Info(f"Added {new_example}")

    return update_user_fig(session_hash)


def change_word_user(examples, session_hash):
    examples = examples.replace(",", " ").split()

    for example in examples:
        if example in user_data[session_hash]["examples"]:
            remove_word_user(example, session_hash)
            add_word_user(example, session_hash)
            gr.Info(f"Changed image for {example}")

    return update_user_fig(session_hash)


def clear_words_user(session_hash):
    if session_hash in user_data:
        while user_data[session_hash]["examples"]:
            remove_word_user(user_data[session_hash]["examples"][-1], session_hash)
        return update_user_fig(session_hash)
    return ""


def set_axis_user(axis_name, which_axis, from_words, to_words, session_hash):
    if axis_name != "residual":
        from_words, to_words = (
            from_words.replace(",", " ").split(),
            to_words.replace(",", " ").split(),
        )
        axis_emb = get_axis_embeddings(from_words, to_words)
        user_data[session_hash]["axis"][axisMap[which_axis]] = axis_emb
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name

        for i, name in enumerate(user_data[session_hash]["axis_names"]):
            if name == "residual":
                user_data[session_hash]["axis"][i] = calculate_residual(
                    user_data[session_hash]["axis"],
                    user_data[session_hash]["axis_names"],
                    from_words,
                    to_words,
                    i,
                )
                user_data[session_hash]["axis_names"][i] = "residual"
    else:
        residual = calculate_residual(
            user_data[session_hash]["axis"],
            user_data[session_hash]["axis_names"],
            residual_axis=axisMap[which_axis],
        )
        user_data[session_hash]["axis"][axisMap[which_axis]] = residual
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name

    user_data[session_hash]["coords"] = (
        get_concat_embeddings(user_data[session_hash]["examples"])
        @ user_data[session_hash]["axis"].T
    )
    user_data[session_hash]["coords"][:, 1] = 5 * (
        1.0 - user_data[session_hash]["coords"][:, 1]
    )

    return update_user_fig(session_hash)


def generate_word_emb_vis(prompt, save_to_file=False, viz_dir=None):
    emb = get_word_embeddings(prompt).reshape(77, 768)[1]

    plt.figure(figsize=(20, 3))
    plt.imshow([emb], cmap="inferno", aspect="auto")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    if save_to_file and viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        safe_filename = "".join([c if c.isalnum() else "_" for c in prompt])
        viz_path = os.path.join(viz_dir, f"{safe_filename}_emb.png")
        try:
            with open(viz_path, "wb") as f:
                f.write(buf.getvalue())
        except Exception as e:
            print(f"Error saving embedding to file: {e}")

    img = "data:image/png;base64, " + base64.b64encode(buf.getvalue()).decode("utf-8")
    return img


def generate_word_embedding_visualization(word, session_hash):
    """Generate and save the word embedding visualization"""
    if not session_hash or not word:
        return None, None, "Invalid session or word"

    try:
        if session_hash not in user_data:
            return None, None, f"Invalid session"

        if word not in user_data[session_hash]["examples"]:
            return None, None, f"Error: '{word}' not in examples"

        examples_dir = get_user_examples_dir(session_hash)
        viz_dir = get_user_viz_dir(session_hash)

        if not examples_dir or not viz_dir:
            return None, None, "Error: Could not create directories"

        str_viz_dir = str(viz_dir)
        emb_viz_b64 = generate_word_emb_vis(
            word, save_to_file=True, viz_dir=str_viz_dir
        )

        emb_viz_bytes = base64.b64decode(emb_viz_b64.split(",")[1])
        emb_viz = Image.open(BytesIO(emb_viz_bytes))

        image_path = examples_dir / f"{get_safe_filename(word)}.jpg"

        if image_path.exists():
            generated_img = Image.open(str(image_path))
        else:
            image = pipe(
                prompt=word,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            image.save(str(image_path), format="JPEG")
            generated_img = image

        return emb_viz, generated_img, f"Visualization for '{word}'"
    except Exception as e:
        print(f"Error generating visualization for '{word}': {e}")
        return None, None, f"Error: {str(e)}"


def load_user_gallery(session_hash):
    """Load the gallery of example images for this user"""
    if not session_hash:
        return []

    if session_hash not in user_data:
        return []

    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return []

    example_images = []
    base_examples_dir = "DiffusionDemo/images/examples"

    for example in user_data[session_hash]["examples"]:
        safe_filename = get_safe_filename(example)
        dest_path = os.path.join(examples_dir, f"{safe_filename}.jpg")
        
        if os.path.exists(dest_path):
            try:
                image = Image.open(dest_path)
                example_images.append((image, example))
                continue
            except Exception as e:
                print(f"Error loading image for '{example}': {e}")
        
        source_path = os.path.join(base_examples_dir, f"{safe_filename}.jpg")
        
        if os.path.exists(source_path):
            try:
                import shutil
                shutil.copy2(source_path, dest_path)
                image = Image.open(dest_path)
                example_images.append((image, example))
                print(f"Copied image for '{example}' from pre-generated source")
            except Exception as e:
                print(f"Error copying image for '{example}': {e}")
        else:
            try:
                image = pipe(
                    prompt=example,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                image.save(dest_path, format="JPEG")
                example_images.append((image, example))
                print(f"Generated new image for '{example}' as pre-generated source was not found")
            except Exception as e:
                print(f"Error generating image for '{example}': {e}")
                continue

    return example_images

__all__ = [
    "generate_user_html",
    "is_new_session",
    "init_user_session",
    "update_user_fig",
    "add_word_user",
    "remove_word_user",
    "add_rem_word_user",
    "change_word_user",
    "clear_words_user",
    "set_axis_user",
    "generate_word_emb_vis",
    "generate_word_embedding_visualization",
    "load_user_gallery",
]