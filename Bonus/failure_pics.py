from datasets import MyDataset
from torchvision import transforms
import torchvision.models as models
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from nets.lenet import LeNet
from nets.my_net import MY_NET
from tqdm import tqdm
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)

# Create a folder to store the failed images
fail_folder = os.path.join(current_directory, "fail", "weighted_boosting_processed")
# Check if the 'fail_folder' exists
if os.path.exists(fail_folder):
    # Iterate through all files in the directory and remove them
    for filename in os.listdir(fail_folder):
        file_path = os.path.join(fail_folder, filename)
        os.remove(file_path)

    # Remove the empty directory
    os.rmdir(fail_folder)

# Recreate the empty 'fail_folder' directory
os.makedirs(fail_folder, exist_ok=True)

# Define data transformation function with common data augmentation techniques
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, saturation=0.2),  # Adjust brightness and saturation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Create datasets and dataloaders
test_dataset = MyDataset(folder_path=parent_directory+"/Dataset_2/", csv_file=parent_directory+"/Dataset_2/Test.csv", transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load images corresponding to predicted labels from the "Meta" folder
meta_images = []  # List to store images for each predicted class
for i in range(len(test_dataset.classes)):
    image_path = os.path.join(parent_directory, "Dataset_2/Meta", f"{i}.png")
    image = plt.imread(image_path)
    meta_images.append(image)


# Set the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = ["alexnet", "lenet", "my_net", "resnet18", "squeezenet1_0", "vgg16", "vit_b_16"]
# model_names = ["resnet18"]
checkpoint_paths = [
    current_directory+"/checkpoints/alexnet_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/lenet_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/my_net_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/resnet18_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/squeezenet1_0_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/vgg16_epoch10_lr0.001_wd0.001.pt",
    current_directory+"/checkpoints/vit_b_16_epoch10_lr0.0001_wd0.001.pt"
]

model_list = []  # stroring the models
model_weights = [0.9854, 0.8856, 0.9736, 0.9759, 0.9696, 0.9675, 0.9869]  # weights for weighted voting
# model_weights = [0.9759]  # weights for weighted voting

# Load and initialize the models
for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
    # Load the model
    if model_name == 'alexnet':
        model = models.alexnet()
    elif model_name == 'lenet':
        model = LeNet()
    elif model_name == 'my_net':
        model = MY_NET()
    elif model_name == 'resnet18':
        model = models.resnet18()
    elif model_name == 'squeezenet1_0':
        model = models.squeezenet1_0()
    elif model_name == 'vgg16':
        model = models.vgg16()
    elif model_name == 'vit_b_16':
        model = models.vit_b_16()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Load the checkpoint  
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    model = model.to(device)
    model_list.append(model)

# Test the models and perform weighted voting
predictions = []
true_labels = []
# Track failed predictions
failed_indices = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader, desc="Testing", leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        # Perform inference with each model and apply the weights
        model_probabilities = []
        for i, model in enumerate(model_list):
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, :43]
            model_probabilities.append(probabilities.cpu().numpy() * model_weights[i])

        # Weighted Voting
        ensemble_probabilities = np.sum(model_probabilities, axis=0)
        ensemble_probabilities /= np.sum(ensemble_probabilities, axis=1, keepdims=True)

        # Select the label with the highest weighted probability for each sample
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)

        # Track predictions and true labels
        predictions.extend(ensemble_predictions)
        true_labels.extend(labels.cpu().numpy())
        # Track failed predictions
        failed_indices.extend((batch_idx * 128 + idx) for idx, (true_label, pred_label) in enumerate(zip(labels.cpu().numpy(), ensemble_predictions)) if true_label != pred_label)

# Calculate test accuracy and F1 score
test_accuracy = accuracy_score(true_labels, predictions)
f1_macro = f1_score(true_labels, predictions, average='macro')
print(f"Weighted Boosting - Test Accuracy: {test_accuracy:.4f} - Test F1-score (macro): {f1_macro:.4f}")

# Save failed images along with their true and predicted labels
for idx in failed_indices:
    image, true_label = test_dataset[idx]
    image = transforms.ToPILImage()(image)  # Convert from tensor to image
    pred_label = predictions[idx]

    # Create a subplot with the original image and the predicted label
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    original_image = test_dataset.get_original_image(idx)
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f"Source Image:{os.path.basename(test_dataset.data[idx][0])}")
    axes[0, 0].axis('off')

    # Create a subplot with the original image and the predicted label
    axes[0,1].imshow(image)
    axes[0,1].set_title(f"PreProcessed Image:{os.path.basename(test_dataset.data[idx][0])}")
    axes[0,1].axis('off')

    # Load the corresponding model name for the predicted label
    true_image = meta_images[true_label]
    axes[1,0].imshow(true_image)
    axes[1,0].set_title(f"True Label: {true_label}")
    axes[1,0].axis('off')

    # Load the corresponding model name for the predicted label
    pred_image = meta_images[pred_label]
    axes[1,1].imshow(pred_image)
    axes[1,1].set_title(f"Predicted Label: {pred_label}")
    axes[1,1].axis('off')

    # Save the comparison plot in the "fail" folder
    plt.savefig(os.path.join(fail_folder, f"fail_{idx}.png"))
    plt.close()