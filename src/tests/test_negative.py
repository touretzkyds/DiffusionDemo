from src.util.params import *
from src.pipelines.negative import *
import matplotlib.pyplot as plt

images, images_neg, _ = display_negative_images(prompt, seed, num_inference_steps, negative_prompt=negative_prompt)

fig, ax = plt.subplots(2, 1, dpi=200, figsize=(1, 2))
fig.suptitle("Without Negative Prompt (Top) vs With Negative Prompt (Bottom)")
ax[0].imshow(images)
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].imshow(images_neg)
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.show()