from src.util.params import *
from src.pipelines.guidance import *
import matplotlib.pyplot as plt

images, _ = display_guidance_images(prompt, seed, num_inference_steps, guidance_values)

fig, ax = plt.subplots(1, len(images), dpi=200, figsize=(len(images), 2))
fig.suptitle("Guidance Scale")

for i in range(len(images)):
    ax[i].imshow(images[i][0])
    ax[i].set_title(images[i][1])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()