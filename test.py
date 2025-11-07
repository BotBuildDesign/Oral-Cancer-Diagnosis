# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 23:43:43 2025

@author: Kwaku Yeboah
"""

import torch
import random
from sklearn.manifold import TSNE
import cv2
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from model import MultiStreamWaveletNet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score 
from sklearn.metrics import precision_score, recall_score


# ---------------------------
# Utility Functions
# ---------------------------

# Display Function
def imshow_batch(img_tensor, labels=None, mean=[0.5]*3, std=[0.5]*3, title=None):
    """
    Display a batch of images after unnormalizing.
    """
    img_tensor = img_tensor.cpu()  # Ensure it's on CPU
    img_tensor = make_grid(img_tensor)  # Merge batch into grid
   # img_tensor = unnormalize(img_tensor, mean, std)  # Unnormalize
    img_tensor = img_tensor.clamp(0, 1)  # Ensure range [0, 1]
    np_img = img_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    plt.savefig("Confusion Matrix.jpg")
    plt.show()

def multiclass_metrics(preds, targets, average='macro'):
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    return precision, recall


# ---------------------------
# Load Dataset
# ---------------------------


test_path = r"E:\Oral_cancer\train"  
batch_size= 32
transform = transforms.Compose([
    transforms.Resize(( 128,128)),
    transforms.ToTensor()
    ])


test_data = datasets.ImageFolder(root=test_path, transform=transform)

# Set the seed for reproducibility
SEED = 64

# Set Python's built-in random seed
random.seed(SEED)

# Set NumPy's random seed
np.random.seed(SEED)

# Set PyTorch's random seed
torch.manual_seed(SEED)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)



# Dataloaders
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)         # test batch size = 1 usually

# Class names
classnames = test_data.classes

# Checking dataset sizes

print(f"Test samples: {len(test_data)}")
print(f'{classnames}')

# Displaying image with labels
# get a batch on images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# display images with labels
imshow_batch(images[:5], labels[:5], title="Validation images")



# ---------------------------
# Load Saved Model
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure num_classes matches your training
model = MultiStreamWaveletNet(in_channels=3, num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()


# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate(model, loader, class_names):
    y_true, y_pred, y_probs = [], [], []
    embeddings, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass (classification)
            outputs = model(images)  # logits
            probs = torch.softmax(outputs, dim=1)[:, 1]  # class-1 prob for binary
            preds = torch.argmax(outputs, dim=1)

            # Forward pass (embeddings)
            embs = model(images, return_embedding=True)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            embeddings.append(embs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    # Concatenate embeddings
    import numpy as np
    embeddings = np.concatenate(embeddings, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # Compute metrics
    accuracy = 100 * (np.array(y_true) == np.array(y_pred)).mean()
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    return accuracy, report, cm, auc, embeddings, labels_all


# ---------------------------
# Run Evaluation
# ---------------------------
accuracy, report, cm, auc, embeddings, labels_all = evaluate(model, test_loader, test_data.classes)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test AUC: {auc:.2f}")
print(report)

# ---------------------------
# Plot Confusion Matrix
# ---------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
#.title("Confusion Matrix")
plt.savefig("Confusion Matrix_SEC")
plt.show()

# ---------------------------
# Plot Embeddings with t-SNE
# ---------------------------

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# Convert numeric labels -> class names
labels_all = labels_all.reshape(-1)  # flatten
label_names = [classnames[int(l)] for l in labels_all]

# Define palette using classnames
palette = {classnames[0]: "red", classnames[1]: "blue"}

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=emb_2d[:, 0],
    y=emb_2d[:, 1],
    hue=label_names,
    palette=palette,
    s=40,
    alpha=0.8,
    edgecolor="k"
)
#plt.title("t-SNE of Learned Embeddings")
plt.legend(title="Classes", loc="lower left")
plt.savefig("t-SNE of Learned Embeddings_SEC.png", dpi=300, bbox_inches="tight")
plt.show()

