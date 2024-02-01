from src.util.params import *
from src.pipelines.perturbations import *
import matplotlib.pyplot as plt

images = display_spread_images(prompt, seed, num_inference_steps, num_images, perturbation_size)
fig, ax = plt.subplots(1, num_images + 1, dpi=200, figsize=(num_images, 2))
fig.suptitle("Spread Images")

for i in range(num_images + 1):
    ax[i].imshow(images[i][0])
    ax[i].set_title(images[i][1])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()