from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image

#url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
#image1 = Image.open(requests.get(url_image1, stream=True).raw)
#url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
#image2 = Image.open(requests.get(url_image2, stream=True).raw)

url_image1 = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/TEST_registr/RPf_00309_INPUT.png'
url_image2 = f'/home/marina/PycharmProjects/RL_puzzle_solver/output/RePAIR_exp_batch3_clean_TEST/TEST_registr/RPf_00309_reference.png'
image1 = Image.open(url_image1)
image2 = Image.open(url_image2)
images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

image_sizes = [[(image.height, image.width) for image in images]]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
for i, output in enumerate(outputs):
    print("For the image pair", i)
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
        )
