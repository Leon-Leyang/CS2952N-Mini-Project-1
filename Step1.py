import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression

# Fix the seed for reproducibility
torch.manual_seed(20)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)

# Download the dataset
train = CIFAR10(root='./data', download=True, train=True, transform=preprocess)
test = CIFAR10(root='./data', download=True, train=False, transform=preprocess)

# Construct the text prompts
prompts = clip.tokenize([f"a photo of a {c}" for c in test.classes]).to(device)

# Sample random subsets
sampled_train = random_split(train, [1000, len(train) - 1000])[0]
sampled_test = random_split(test, [500, len(test)-500])[0]

# Get the test labels
test_labels = [label for _, label in sampled_test]


def get_zero_shot_predictions(dataset):
    predictions = []

    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=500):
            images = images.to(device)
            img_logits, _ = model(images, prompts)
            predictions.extend(img_logits.argmax(dim=-1).cpu().numpy())

    return predictions


# Get the zero-shot predictions
zero_shot_predictions = get_zero_shot_predictions(sampled_test)

# Calculate the accuracy
accuracy = (np.array(zero_shot_predictions) == np.array(test_labels)).mean() * 100
print(f'Zero-shot accuracy: {accuracy:.2f}%')


# Perform linear probing
def get_features(dataset):
    """
    Extract features from the dataset using the CLIP model.

    Source: https://github.com/openai/CLIP#modelimage-tensor-text-tensor
    """
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=100):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Source: https://github.com/openai/CLIP#modelimage-tensor-text-tensor
# Calculate the image features
train_features, train_labels = get_features(sampled_train)
test_features, test_labels = get_features(sampled_test)
# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
classifier.fit(train_features, train_labels)
# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f'Linear probing accuracy: {accuracy:.2f}%')
