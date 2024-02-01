from src.util.params import *
from src.pipelines.circular import *
import matplotlib.pyplot as plt

images = display_circular_images(prompt, seed, num_inference_steps, num_images, degree)

fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
fig.suptitle("Circular Images")

for i in range(num_images):
    ax[i].imshow(images[i][0])
    ax[i].set_title(images[i][1])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()