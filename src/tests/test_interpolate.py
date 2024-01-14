from src.util.params import *
from src.pipelines.interpolate import *
import matplotlib.pyplot as plt   

images = display_interpolate_images(seed, promptA, promptB, num_inference_steps, num_images)
fig, ax = plt.subplots(1, num_images, dpi=200, figsize=(num_images, 2))
fig.suptitle("Interpolate Images")

for i in range(num_images):
    ax[i].imshow(images[i][0])
    ax[i].set_xticks([])
    ax[i].set_yticks([])

ax[0].set_title("Prompt A")
ax[num_images-1].set_title("Prompt B") 

plt.show()