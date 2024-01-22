from src.util.params import *
from src.pipelines.seed import *
import matplotlib.pyplot as plt

images = display_seed_images(prompt, num_inference_steps, num_images)

fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
fig.suptitle("Multiple Seeds")

for i in range(num_images):
    ax[i].imshow(images[i][0])
    ax[i].set_title(images[i][1])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()