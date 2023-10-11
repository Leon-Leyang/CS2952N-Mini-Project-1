import os
import clip
import torch
import requests
from PIL import Image


# Fix the seed for reproducibility
torch.manual_seed(50)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('RN50', device)

# Construct the text prompts from the ImageNet classes
URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(URL)
imagenet_class_index = response.json()
all_classes = [item[1] for item in imagenet_class_index.values()]
init_class_prompts = clip.tokenize([f"the photo contains {c}" for c in all_classes]).to(device)
follow_up_prompt_template = "the photo, besides {}, contains {}"
end_prompt_template = "the photo, besides {}, contains no other classes of objects"


# Get the class predictions
def get_class_predictions(img, num_classes=5):
    candidate_classes = []
    with torch.no_grad():
        img = img.to(device)
        img_logits, _ = clip_model(img, init_class_prompts)

        # Get the one class with the highest logit score
        best_class_idx = img_logits.argmax(dim=-1).cpu().numpy()[0]
        best_class = all_classes[best_class_idx]
        candidate_classes.append(best_class)

        # Get the remaining classes using the follow-up prompt
        for _ in range(num_classes - 1):
            current_class_candidates = [cls for cls in all_classes if cls not in candidate_classes]
            follow_up_prompt = [follow_up_prompt_template.format(", ".join(candidate_classes), cls) for cls in
                                current_class_candidates]
            end_prompt = end_prompt_template.format(", ".join(candidate_classes))
            follow_up_prompt.append(end_prompt)
            follow_up_prompt = clip.tokenize(follow_up_prompt).to(device)
            img_logits, _ = clip_model(img, follow_up_prompt)
            best_class_idx = img_logits.argmax(dim=-1).cpu().numpy()[0]
            if best_class_idx == len(follow_up_prompt) - 1:
                print("No more classes to add")
                break
            best_class = current_class_candidates[best_class_idx]
            candidate_classes.append(best_class)
    return candidate_classes


# Get a random image from ./data/Flickr8k_Dataset
def get_random_image():
    img_dir = './data/Flicker8k_Dataset'
    image_files = os.listdir(img_dir)
    image_file = image_files[torch.randint(len(image_files), size=(1,)).item()]
    image_path = os.path.join(img_dir, image_file)
    return image_file, preprocess(Image.open(image_path)).unsqueeze(0)


# Get the top k predictions for a random image
num_classes = 5
img_file, img = get_random_image()
top_k_classes = get_class_predictions(img, num_classes)
print(f"Top {num_classes} predictions for {img_file}:")
print(top_k_classes)



