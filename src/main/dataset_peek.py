import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from clip_retrieval.clip_client import ClipClient, Modality

def retrieve_images(prompt):
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion5B-L-14",
        aesthetic_score=9,
        aesthetic_weight=0.5,
        modality=Modality.IMAGE,
        num_images=10,
    )
    results = client.query(text=prompt)
    image_captions = []
    for result in results:
        caption, url = result["caption"], result["url"]
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            image_captions.append((image, caption))
        except:
            pass
    return image_captions
     
if __name__ == "__main__":
    prompt = "cat"   
    images = retrieve_images(prompt)

    os.makedirs("./research/production/src/output/dataset_peek", exist_ok=True)
    for i in range(len(images)):
        plt.title(images[i][1], wrap=True)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"./research/production/src/output/dataset_peek/{prompt}_{i}.png")
        plt.imshow(images[i][0])