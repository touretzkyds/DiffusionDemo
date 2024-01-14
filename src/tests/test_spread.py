from src.util.params import *
from src.pipelines.spread import *
import matplotlib.pyplot as plt

images = display_spread_images(prompt, seed, num_inference_steps, num_images, differentiation)
fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
fig.suptitle("Spread Images")

for i in range(num_images):
    ax[i].imshow(images[i][0])
    ax[i].set_title(images[i][1])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()