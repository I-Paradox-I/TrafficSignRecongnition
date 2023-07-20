from datasets import MyDataset
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)


# Function to plot original and preprocessed images
def plot_images(original_img, preprocessed_img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot preprocessed image
    axes[1].imshow(preprocessed_img)
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')

    plt.savefig(parent_directory + "/Bonus/Preprocessed_Image.png")

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

# Get the original image directly from the file
original_image_path = parent_directory + "/Dataset_2/Test/12474.png"
original_image = Image.open(original_image_path)

# Get the preprocessed image from the dataloader
preprocessed_image = test_dataset[12474][0]

# Convert the numpy array to a PIL image for visualization
preprocessed_image = transforms.ToPILImage()(preprocessed_image)

# Plot the images
plot_images(original_image, preprocessed_image)