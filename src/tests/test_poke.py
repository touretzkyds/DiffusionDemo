from src.util.params import *
from src.pipelines.poke import *
import matplotlib.pyplot as plt

images, modImages = display_poke_images(prompt, seed, num_inference_steps, poke=True, pokeX=pokeX, pokeY=pokeY, pokeHeight=pokeHeight, pokeWidth=pokeWidth, intermediate=intermediate)

if intermediate:
    fig, ax = plt.subplots(2, num_inference_steps, dpi=200, figsize=(num_inference_steps, 3))
    fig.suptitle("Original (Top) vs Poked (Bottom)")

    for i in range(num_inference_steps):
        ax[0][i].set_title(f"Step {i+1}")
        ax[0][i].imshow(images[i])
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])

        ax[1][i].set_title(f"Step {i+1}")
        ax[1][i].imshow(modImages[i])
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])

else:
    fig, ax = plt.subplots(2, 1, dpi=200, figsize=(1, 2))
    fig.suptitle("Original (Top) vs Poked (Bottom)")
    ax[0].imshow(images)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(modImages)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

plt.show()